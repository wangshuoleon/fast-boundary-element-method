function stokes_flow_sphere_BEM()
    % Parameters
    tic
    R = 1.0;           % Radius of the sphere
    U0 = [0, 0, 1];    % Constant velocity of the sphere [Ux, Uy, Uz]
    N = 1000;          % Number of surface elements (increased for accuracy)
    mu = 1.0;          % Dynamic viscosity of the fluid
    delta = R / 10;  % Small offset to avoid singularity
    epsilon = 1e-8;    % Regularization term for diagonal

    % Discretize the sphere surface
    [vertices,faces] = generateSphereMesh(R, N);
    
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

function [vertices,faces] = generateSphereMesh(R, N)
    % Generate a spherical mesh using MATLAB's built-in function
    [x, y, z] = sphere(round(sqrt(N/2))); % Adjust resolution to get ~N elements
    
     vertices = R * [x(:), y(:), z(:)];
    %  [vertices, ~, ~] = unique(vertices, 'rows', 'stable');
      [m, ~] = size(x);
    faces = [];

    % Build all faces first (same as original)
    % Quadrilateral faces (split into triangles)
    for i = 1:m-1
        for j = 1:m-1
            v1 = (i-1)*m + j;
            v2 = (i-1)*m + j+1;
            v3 = i*m + j+1;
            v4 = i*m + j;
            faces = [faces; v1, v2, v3; v1, v3, v4];
        end
    end

    % Pole triangles
    northPole = 1;
    for j = 1:m-1
        v2 = 1 + j;
        v3 = 1 + j + 1;
        faces = [faces; northPole, v2, v3];
    end

    southPole = m*m;
    for j = 1:m-1
        v2 = (m-1)*m + j;
        v3 = (m-1)*m + j + 1;
        faces = [faces; southPole, v3, v2];
    end

    % --- Critical Fix: Remove duplicate vertices and update faces ---
    [vertices, ~, ic] = unique(vertices, 'rows', 'stable');
    
    % Update face indices using the mapping from unique()
    faces = ic(faces);
    
    % Optional: Verify no degenerate faces remain
    validFaces = all(diff(faces, 1, 2) ~= 0, 2); % Check for v1 ¡Ù v2 ¡Ù v3
    faces = faces(validFaces, :);
    %  faces = delaunay(vertices(:,1), vertices(:,2), vertices(:,3)); % Triangulate the surface
end

function [points, weights] = get_triangle_quadrature()
% 7-point Gaussian quadrature for triangles
points = [...
    0.1012865073235, 0.1012865073235; ...
    0.7974269853531, 0.1012865073235; ...
    0.1012865073235, 0.7974269853531; ...
    0.4701420641051, 0.0597158717898; ...
    0.4701420641051, 0.4701420641051; ...
    0.0597158717898, 0.4701420641051; ...
    0.3333333333333, 0.3333333333333];
weights = [...
    0.1259391805448; ...
    0.1259391805448; ...
    0.1259391805448; ...
    0.1323941527885; ...
    0.1323941527885; ...
    0.1323941527885; ...
    0.2250000000000];
end
