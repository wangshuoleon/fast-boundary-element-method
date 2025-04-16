function stokes_flow_sphere_BEM()
    % Parameters
    tic
    R = 1.0;           % Radius of the sphere
    U0 = [0, 0, 1];    % Constant velocity of the sphere [Ux, Uy, Uz]
    N = 1000;          % Number of surface elements (increased for accuracy)
    mu = 1.0;          % Dynamic viscosity of the fluid
    delta = R / 5;  % Small offset to avoid singularity
    epsilon = 1e-8;    % Regularization term for diagonal

    % Discretize the sphere surface
    [vertices] = generateSphereMesh(R, N);
    
   [vertices, ~, ic] = unique(vertices, 'rows', 'stable');
 %  faces = delaunay(x(:), y(:), z(:)); % Triangulate the surface
    
   % node_normals = computeNodeNormals(vertices, faces);
    
    % Plot the mesh
% figure;
% trisurf(faces, vertices(:, 1), vertices(:, 2), vertices(:, 3));
% hold on;
% 
% % Plot the normal vectors
% quiver3(vertices(:, 1), vertices(:, 2), vertices(:, 3), ...
%         node_normals(:, 1), node_normals(:, 2), node_normals(:, 3), 'r');
% title('Mesh with Node Normals');
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% axis equal;
% hold off;
    % Number of nodes
    numNodes = size(vertices, 1);

    % Initialize the system matrix and right-hand side vector
    A = zeros(3 * numNodes, 3 * numNodes); % 3D problem (x, y, z)
    b = zeros(3 * numNodes, 1);
tic
    % Compute the influence matrix
    for i = 1:numNodes
        % Observation point: slightly outside the sphere
        obs_point = vertices(i, :) ;

        for j = 1:numNodes
            % Source point: underneath the sphere surface
            src_point = vertices(j, :)*(1-delta/R);

            % Distance vector and magnitude
            r_vec = obs_point - src_point;
            r = norm(r_vec);

            % Stokeslet kernel
            G = (eye(3) / r + (r_vec' * r_vec) / r^3) / (8 * pi * mu);

            % Populate the influence matrix
            A((3*i-2):(3*i), (3*j-2):(3*j)) = G;
        end

        % Add regularization term to diagonal
        A((3*i-2):(3*i), (3*i-2):(3*i)) = A((3*i-2):(3*i), (3*i-2):(3*i)) + epsilon * eye(3);

        % Right-hand side: velocity boundary condition
        b((3*i-2):(3*i)) = U0';
    end

   toc
   tic
    % Solve for the force distribution using preconditioned system
    f = A\ b;
   toc
   % Move data to GPU
% A_gpu = gpuArray(A);
% b_gpu = gpuArray(b);
% 
% % Solve using cuSOLVER (LU factorization)
% tic;
% x_gpu = A_gpu \ b_gpu; % MATLAB's backslash operator uses cuSOLVER on GPU
% toc;
   
    % Check for NaNs or Infs in the solution
    if any(isnan(f)) || any(isinf(f))
        error('Solver returned NaN or Inf. Check the system matrix and regularization.');
    end

    % Reshape the force distribution into a 3D array
    f = reshape(f, [3, numNodes])';

    % Compute the total force
    total_force = sum(f, 1); % Sum over all nodes

    % Analytical solution (Stokes' law)
    analytical_force = 6 * pi * mu * R * U0;

    % Display results
    fprintf('Computed total force: [%.6f, %.6f, %.6f]\n', total_force);
    fprintf('Analytical total force: [%.6f, %.6f, %.6f]\n', analytical_force);

    % Visualize the force distribution (x-component)
    figure;
    scatter3(vertices(:, 1), vertices(:, 2), vertices(:, 3), 50, f(:, 3), 'filled');
    colorbar;
    title('Force Distribution on Sphere (z-component)');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    axis equal;
    keyboard
end

function [vertices] = generateSphereMesh(R, N)
    % Generate a spherical mesh using MATLAB's built-in function
    [x, y, z] = sphere(round(sqrt(N/2))); % Adjust resolution to get ~N elements
    
     vertices = R * [x(:), y(:), z(:)];
     [vertices, ~, ~] = unique(vertices, 'rows', 'stable');
    %  faces = delaunay(vertices(:,1), vertices(:,2), vertices(:,3)); % Triangulate the surface
end

function node_normals = computeNodeNormals(vertices, faces)
    % vertices: N x 3 matrix of vertex coordinates
    % faces: M x 3 matrix of face connectivity (triangular faces)
    % node_normals: N x 3 matrix of normal vectors at each node

    % Initialize node normals
    num_nodes = size(vertices, 1);
    node_normals = zeros(num_nodes, 3);

    % Compute face normals
    face_normals = zeros(size(faces, 1), 3);
    for i = 1:size(faces, 1)
        % Get the vertices of the current face
        v1 = vertices(faces(i, 1), :);
        v2 = vertices(faces(i, 2), :);
        v3 = vertices(faces(i, 3), :);

        % Compute edge vectors
        edge1 = v2 - v1;
        edge2 = v3 - v1;

        % Compute face normal using cross product
        face_normals(i, :) = cross(edge1, edge2);
    end

    % Average face normals at each node
    for i = 1:num_nodes
        % Find all faces that share this node
        [shared_faces, ~] = find(faces == i);

        % Average the normals of the shared faces
        node_normals(i, :) = mean(face_normals(shared_faces, :), 1);

        % Normalize the normal vector
        node_normals(i, :) = node_normals(i, :) / norm(node_normals(i, :));
    end
end