function beamStressAnalysis()
%BEAMSTRESSANALYSIS Interactive beam stress/strain analysis app
%   Uses PDE Toolbox for structural analysis and VTK.js for visualization
%
%   This app allows users to:
%   - Define beam geometry (length, width, height)
%   - Set material properties (Young's modulus, Poisson's ratio, density)
%   - Apply boundary conditions (fixed, pinned, roller supports)
%   - Apply loads (point forces, distributed loads, pressure)
%   - Visualize stress and displacement results in 3D
%
%   Example:
%       beamStressAnalysis()

    % Create main figure
    fig = uifigure('Name', 'Beam Stress Analysis - PDE Toolbox', ...
                   'Position', [50 50 1400 900], ...
                   'Color', [0.1 0.1 0.15], ...
                   'CloseRequestFcn', @onClose);

    % Create HTML component (full window)
    h = uihtml(fig, 'Position', [0 0 1400 900]);
    h.HTMLSource = fullfile(pwd, 'beamAnalysis.html');

    % Store app data
    appData = struct();
    appData.model = [];
    appData.results = [];
    appData.geometry = struct('length', 1, 'width', 0.1, 'height', 0.05);

    % Store in figure
    fig.UserData = appData;

    % Set up event handler
    h.HTMLEventReceivedFcn = @(src, event) handleEvent(src, event, fig);

    function onClose(src, ~)
        delete(src);
    end
end

function handleEvent(src, event, fig)
%HANDLEEVENT Process events from JavaScript frontend

    eventName = event.HTMLEventName;
    eventData = event.HTMLEventData;

    try
        switch eventName
            case 'CreateBeam'
                createBeamGeometry(src, eventData, fig);

            case 'RunAnalysis'
                runStructuralAnalysis(src, eventData, fig);

            case 'ChangeResultType'
                changeResultDisplay(src, eventData, fig);

            case 'ExportResults'
                exportAnalysisResults(fig);

            otherwise
                fprintf('Unknown event: %s\n', eventName);
        end

    catch ME
        fprintf('Error in %s: %s\n', eventName, ME.message);
        fprintf('Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
        sendEventToHTMLSource(src, 'Error', ME.message);
    end
end

function createBeamGeometry(src, data, fig)
%CREATEBEAMGEOMETRY Create beam geometry using PDE Toolbox

    % Extract dimensions
    L = data.length;  % Length (x direction)
    W = data.width;   % Width (y direction)
    H = data.height;  % Height (z direction)

    % Store geometry
    appData = fig.UserData;
    appData.geometry = struct('length', L, 'width', W, 'height', H);
    fig.UserData = appData;

    fprintf('Beam geometry created: %.3f x %.3f x %.3f m\n', L, W, H);
    sendEventToHTMLSource(src, 'BeamCreated', struct('success', true));
end

function runStructuralAnalysis(src, params, fig)
%RUNSTRUCTURALANALYSIS Run structural analysis using PDE Toolbox

    fprintf('\n=== Starting Structural Analysis ===\n');

    % Extract parameters
    geom = params.geometry;
    mat = params.material;
    bcs = params.boundaryConditions;
    loads = params.loads;
    meshSizeStr = params.meshSize;

    L = geom.length;
    W = geom.width;
    H = geom.height;

    fprintf('Geometry: L=%.3f, W=%.3f, H=%.3f m\n', L, W, H);
    fprintf('Material: E=%.2e Pa, nu=%.3f, rho=%.1f kg/m3\n', mat.E, mat.nu, mat.rho);

    % Create structural model
    model = femodel(AnalysisType="structuralStatic");

    % Create 3D beam geometry using multicuboid
    gm = multicuboid(L, W, H);

    % Translate to start at origin
    gm = translate(gm, [L/2, W/2, H/2]);

    model.Geometry = gm;

    fprintf('Geometry created with %d faces\n', gm.NumFaces);

    % Assign material properties
    model.MaterialProperties = materialProperties( ...
        YoungsModulus=mat.E, ...
        PoissonsRatio=mat.nu, ...
        MassDensity=mat.rho);

    % Apply boundary conditions
    fprintf('Applying %d boundary conditions...\n', length(bcs));
    for i = 1:length(bcs)
        bc = bcs(i);
        faceID = getFaceID(bc.face, L, W, H, gm);

        if ~isempty(faceID)
            switch bc.type
                case 'fixed'
                    % Fixed: all displacements = 0
                    model.FaceBC(faceID) = faceBC(Constraint="fixed");
                    fprintf('  Applied fixed BC on face %d (%s)\n', faceID, bc.face);

                case 'pinned'
                    % Pinned: displacements = 0, rotations free
                    model.FaceBC(faceID) = faceBC(Constraint="fixed");
                    fprintf('  Applied pinned BC on face %d (%s)\n', faceID, bc.face);

                case 'roller'
                    % Roller: allow sliding in one direction
                    model.FaceBC(faceID) = faceBC(YDisplacement=0, ZDisplacement=0);
                    fprintf('  Applied roller BC on face %d (%s)\n', faceID, bc.face);
            end
        end
    end

    % Apply loads
    fprintf('Applying %d loads...\n', length(loads));
    for i = 1:length(loads)
        load = loads(i);
        faceID = getFaceID(load.face, L, W, H, gm);

        if ~isempty(faceID)
            % Get face area for converting force to pressure
            faceArea = getFaceArea(load.face, L, W, H);

            switch load.type
                case 'point'
                    % Apply as concentrated force (use SurfaceTraction with small area approximation)
                    % For simplicity, distribute over the face
                    px = load.fx / faceArea;
                    py = load.fy / faceArea;
                    pz = load.fz / faceArea;
                    model.FaceLoad(faceID) = faceLoad(SurfaceTraction=[px; py; pz]);
                    fprintf('  Applied point load as traction [%.1f, %.1f, %.1f] Pa on face %d\n', px, py, pz, faceID);

                case 'distributed'
                    % Distributed load over the face
                    px = load.fx / faceArea;
                    py = load.fy / faceArea;
                    pz = load.fz / faceArea;
                    model.FaceLoad(faceID) = faceLoad(SurfaceTraction=[px; py; pz]);
                    fprintf('  Applied distributed load [%.1f, %.1f, %.1f] Pa on face %d\n', px, py, pz, faceID);

                case 'pressure'
                    % Pressure normal to face
                    pressure = sqrt(load.fx^2 + load.fy^2 + load.fz^2) / faceArea;
                    model.FaceLoad(faceID) = faceLoad(Pressure=pressure);
                    fprintf('  Applied pressure %.1f Pa on face %d\n', pressure, faceID);
            end
        end
    end

    % Generate mesh
    meshSize = getMeshSize(meshSizeStr, L, W, H);
    fprintf('Generating mesh (target size: %.4f m)...\n', meshSize);
    model = generateMesh(model, Hmax=meshSize);

    mesh = model.Geometry.Mesh;
    numNodes = size(mesh.Nodes, 2);
    numElements = size(mesh.Elements, 2);
    fprintf('Mesh generated: %d nodes, %d elements\n', numNodes, numElements);

    % Send mesh info to frontend
    sendEventToHTMLSource(src, 'MeshGenerated', struct('numNodes', numNodes, 'numElements', numElements));

    % Solve the problem
    fprintf('Solving structural problem...\n');
    tic;
    results = solve(model);
    solveTime = toc;
    fprintf('Solution completed in %.2f seconds\n', solveTime);

    % Extract results
    displacement = results.Displacement;
    stress = results.VonMisesStress;

    % Calculate displacement magnitude
    dispMag = sqrt(displacement.ux.^2 + displacement.uy.^2 + displacement.uz.^2);
    maxDisp = max(dispMag);
    maxStress = max(stress);
    minStress = min(stress);

    fprintf('Max displacement: %.6f m (%.4f mm)\n', maxDisp, maxDisp*1000);
    fprintf('Max von Mises stress: %.2e Pa (%.2f MPa)\n', maxStress, maxStress/1e6);

    % Store results
    appData = fig.UserData;
    appData.model = model;
    appData.results = results;
    fig.UserData = appData;

    % Prepare data for JavaScript visualization
    nodes = mesh.Nodes;
    elements = mesh.Elements;

    % Get boundary triangles for surface visualization
    [tri, ~] = getBoundaryTriangles(mesh);

    % Flatten arrays for JavaScript
    nodesFlat = reshape(nodes, 1, []);
    dispFlat = [displacement.ux'; displacement.uy'; displacement.uz'];
    dispFlat = reshape(dispFlat, 1, []);

    % Interpolate stress to nodes for smooth visualization
    nodeStress = interpolateStressToNodes(stress, mesh);

    % Send results to frontend
    resultData = struct( ...
        'nodes', nodesFlat, ...
        'elements', reshape(elements, 1, []), ...
        'triangles', reshape(tri, 1, []), ...
        'displacement', dispFlat, ...
        'nodeStress', nodeStress, ...
        'vonMisesStress', stress, ...
        'maxDisplacement', maxDisp, ...
        'maxVonMises', maxStress, ...
        'minVonMises', minStress, ...
        'numNodes', numNodes, ...
        'numElements', numElements, ...
        'solveTime', solveTime ...
    );

    sendEventToHTMLSource(src, 'AnalysisComplete', resultData);
    fprintf('Results sent to visualization\n');
    fprintf('=== Analysis Complete ===\n\n');
end

function faceID = getFaceID(faceName, L, W, H, gm)
%GETFACEID Get face ID from face name

    % For a cuboid created with multicuboid and translated:
    % Face ordering depends on the geometry creation
    % We need to find faces by their position

    faceID = [];

    % Get face centers to identify faces
    for f = 1:gm.NumFaces
        try
            % Get face vertices
            verts = gm.Vertices;
            faceVerts = gm.faceVertices{f};
            faceCoords = verts(faceVerts, :);
            center = mean(faceCoords, 1);

            tol = 1e-6;
            switch faceName
                case 'left'
                    if abs(center(1)) < tol
                        faceID = f;
                        return;
                    end
                case 'right'
                    if abs(center(1) - L) < tol
                        faceID = f;
                        return;
                    end
                case 'front'
                    if abs(center(2)) < tol
                        faceID = f;
                        return;
                    end
                case 'back'
                    if abs(center(2) - W) < tol
                        faceID = f;
                        return;
                    end
                case 'bottom'
                    if abs(center(3)) < tol
                        faceID = f;
                        return;
                    end
                case 'top'
                    if abs(center(3) - H) < tol
                        faceID = f;
                        return;
                    end
            end
        catch
            % If face analysis fails, use default mapping
        end
    end

    % Fallback: use typical cuboid face ordering
    if isempty(faceID)
        switch faceName
            case 'left'
                faceID = 1;
            case 'right'
                faceID = 2;
            case 'front'
                faceID = 3;
            case 'back'
                faceID = 4;
            case 'bottom'
                faceID = 5;
            case 'top'
                faceID = 6;
        end
    end
end

function area = getFaceArea(faceName, L, W, H)
%GETFACEAREA Get face area from face name

    switch faceName
        case {'left', 'right'}
            area = W * H;
        case {'front', 'back'}
            area = L * H;
        case {'bottom', 'top'}
            area = L * W;
        otherwise
            area = L * W; % Default
    end
end

function meshSize = getMeshSize(sizeStr, L, W, H)
%GETMESHSIZE Get mesh size from string

    minDim = min([L, W, H]);

    switch sizeStr
        case 'coarse'
            meshSize = minDim / 2;
        case 'medium'
            meshSize = minDim / 4;
        case 'fine'
            meshSize = minDim / 8;
        case 'veryfine'
            meshSize = minDim / 16;
        otherwise
            meshSize = minDim / 4;
    end
end

function [tri, triNodes] = getBoundaryTriangles(mesh)
%GETBOUNDARYTRIANGLES Extract boundary triangles from tetrahedral mesh

    elements = mesh.Elements;
    nodes = mesh.Nodes;

    % For tetrahedral elements, extract all faces
    % Each tetrahedron has 4 triangular faces
    numElements = size(elements, 2);

    % Face definitions for tetrahedron (node indices within element)
    faceNodes = [1 2 3; 1 2 4; 1 3 4; 2 3 4];

    % Collect all faces
    allFaces = zeros(numElements * 4, 3);
    for i = 1:numElements
        elem = elements(:, i);
        for f = 1:4
            idx = (i-1)*4 + f;
            allFaces(idx, :) = sort(elem(faceNodes(f, :)));
        end
    end

    % Find boundary faces (appear only once)
    [uniqueFaces, ~, ic] = unique(allFaces, 'rows');
    counts = accumarray(ic, 1);
    boundaryIdx = counts == 1;
    boundaryFaces = uniqueFaces(boundaryIdx, :);

    % Need to get original ordering for proper normals
    tri = zeros(size(boundaryFaces));
    for i = 1:size(boundaryFaces, 1)
        sortedFace = boundaryFaces(i, :);
        % Find original face in allFaces
        for j = 1:size(allFaces, 1)
            if isequal(sort(allFaces(j, :)), sortedFace)
                % Get original element and face
                elemIdx = ceil(j/4);
                faceIdx = mod(j-1, 4) + 1;
                elem = elements(:, elemIdx);
                tri(i, :) = elem(faceNodes(faceIdx, :));
                break;
            end
        end
    end

    triNodes = nodes;
end

function nodeStress = interpolateStressToNodes(elementStress, mesh)
%INTERPOLATESTRESSTONODES Interpolate element stress values to nodes

    nodes = mesh.Nodes;
    elements = mesh.Elements;
    numNodes = size(nodes, 2);
    numElements = size(elements, 2);

    % Accumulate stress contributions to each node
    nodeStressSum = zeros(numNodes, 1);
    nodeCount = zeros(numNodes, 1);

    for i = 1:numElements
        elem = elements(:, i);
        stress = elementStress(i);
        for j = 1:length(elem)
            nodeIdx = elem(j);
            nodeStressSum(nodeIdx) = nodeStressSum(nodeIdx) + stress;
            nodeCount(nodeIdx) = nodeCount(nodeIdx) + 1;
        end
    end

    % Average
    nodeStress = nodeStressSum ./ max(nodeCount, 1);
    nodeStress = nodeStress';
end

function changeResultDisplay(src, resultType, fig)
%CHANGERESULTDISPLAY Change the displayed result type

    appData = fig.UserData;
    if isempty(appData.results)
        return;
    end

    results = appData.results;
    mesh = appData.model.Geometry.Mesh;

    switch resultType
        case 'vonMises'
            data = results.VonMisesStress;
            nodeData = interpolateStressToNodes(data, mesh);
            unit = 'Pa';
        case 'displacement'
            disp = results.Displacement;
            nodeData = sqrt(disp.ux.^2 + disp.uy.^2 + disp.uz.^2)';
            unit = 'm';
        case 'dispX'
            nodeData = results.Displacement.ux';
            unit = 'm';
        case 'dispY'
            nodeData = results.Displacement.uy';
            unit = 'm';
        case 'dispZ'
            nodeData = results.Displacement.uz';
            unit = 'm';
        otherwise
            return;
    end

    resultData = struct( ...
        'nodeStress', nodeData, ...
        'minValue', min(nodeData), ...
        'maxValue', max(nodeData), ...
        'unit', unit ...
    );

    sendEventToHTMLSource(src, 'ResultTypeChanged', resultData);
end

function exportAnalysisResults(fig)
%EXPORTANALYSISRESULTS Export analysis results to file

    appData = fig.UserData;
    if isempty(appData.results)
        return;
    end

    % Create results table
    results = appData.results;
    mesh = appData.model.Geometry.Mesh;
    nodes = mesh.Nodes;

    % Export to MAT file
    filename = fullfile(pwd, 'analysis_results.mat');
    save(filename, 'results', 'nodes', '-v7.3');
    fprintf('Results exported to: %s\n', filename);

    % Also export summary to text file
    summaryFile = fullfile(pwd, 'analysis_summary.txt');
    fid = fopen(summaryFile, 'w');
    fprintf(fid, 'Beam Stress Analysis Summary\n');
    fprintf(fid, '============================\n\n');
    fprintf(fid, 'Geometry:\n');
    fprintf(fid, '  Length: %.4f m\n', appData.geometry.length);
    fprintf(fid, '  Width: %.4f m\n', appData.geometry.width);
    fprintf(fid, '  Height: %.4f m\n', appData.geometry.height);
    fprintf(fid, '\nMesh:\n');
    fprintf(fid, '  Nodes: %d\n', size(nodes, 2));
    fprintf(fid, '  Elements: %d\n', size(mesh.Elements, 2));
    fprintf(fid, '\nResults:\n');

    disp = results.Displacement;
    dispMag = sqrt(disp.ux.^2 + disp.uy.^2 + disp.uz.^2);
    fprintf(fid, '  Max Displacement: %.6e m (%.4f mm)\n', max(dispMag), max(dispMag)*1000);
    fprintf(fid, '  Max von Mises Stress: %.6e Pa (%.2f MPa)\n', max(results.VonMisesStress), max(results.VonMisesStress)/1e6);

    fclose(fid);
    fprintf('Summary exported to: %s\n', summaryFile);
end
