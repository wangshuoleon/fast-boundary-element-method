function stokes_flow_sphere_BEM()
    % Parameters
    tic
    R = 1.0;           % Radius of the sphere
    U0 = [0, 0, 1];    % Constant velocity of the sphere [Ux, Uy, Uz]
    N = 1000;          % Number of surface elements (increased for accuracy)
    mu = 1.0;          % Dynamic viscosity of the fluid
    separation = [0,0,5];  % Small offset to avoid singularity
  %   epsilon = 1e-8;    % Regularization term for diagonal

    % Discretize the sphere surface
    [vertices,faces] = generateSphereMesh(R, N);
    % generation of second sphere

    faces=[faces;faces+size(vertices,1)];
    theta=pi/2;
    phi=0;
     R_phi = [cos(phi), -sin(phi), 0; 
             sin(phi), cos(phi), 0; 
             0, 0, 1];
    R_theta = [cos(theta), 0, sin(theta); 
               0, 1, 0; 
               -sin(theta), 0, cos(theta)];
    rotation_matrix =  R_phi*R_theta;
    
    vertices2 = (rotation_matrix * vertices.').' + separation; % Rotate and translate
    vertices=[vertices;vertices2];
   
    
    % Number of nodes
    numNodes = size(vertices, 1);

    % Initialize the system matrix and right-hand side vector
    A = zeros(3 * numNodes, 3 * numNodes); % 3D problem (x, y, z)
    B = zeros(3 * numNodes, 1);
    % Precompute triangle quadrature points and weights
    [quad_points, quad_weights] = get_triangle_quadrature();
    num_quads = length(quad_weights);

    % Loop over triangle faces (observation domain)
    for tri = 1:size(faces,1)
        nodes = faces(tri, :);
        v1 = vertices(nodes(1), :);
        v2 = vertices(nodes(2), :);
        v3 = vertices(nodes(3), :);

        % Triangle area and edge vectors
        area = 0.5 * norm(cross(v2 - v1, v3 - v1));

        % Loop over each quadrature point
        for q = 1:num_quads
            % Barycentric coords -> physical position
            a = quad_points(q,1);
            b = quad_points(q,2);
            c = 1 - a - b;
            obs_point = a*v1 + b*v2 + c*v3;

            % Shape function contributions (for redistribution)
            shape = [c, a, b];

            % Evaluate contribution of all nodal forces to this quadrature point
            for src_node = 1:numNodes
                src_point = vertices(src_node,:) ;
                r_vec = obs_point - src_point;
                r = norm(r_vec);

                % Green's function (Stokeslet)
                G = (eye(3)/r + (r_vec'*r_vec)/r^3) / (8*pi*mu);

                % Distribute influence to triangle nodes via shape functions
                for k = 1:3
                    row_idx = (nodes(k)-1)*3 + (1:3);    % target row in A
                    col_idx = (src_node-1)*3 + (1:3);    % source force node

                    % Contribution to A matrix
                    A(row_idx, col_idx) = A(row_idx, col_idx) + ...
                        shape(k) * G * quad_weights(q) * area;
                end
            end

            % Distribute RHS velocity to triangle nodes
            for k = 1:3
                idx = (nodes(k)-1)*3 + (1:3);
                B(idx) = B(idx) + shape(k) * U0' * quad_weights(q) * area;
            end
        end
    end

    % Regularize diagonal blocks
%     for i = 1:numNodes
%         idx = (3*i-2):(3*i);
%         A(idx, idx) = A(idx, idx) + epsilon * eye(3);
%     end

   tic
    % Solve for the force distribution using preconditioned system
    f = A\ B;
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
% 4-point Gaussian quadrature for triangles (degree of precision = 3)
points = [...
    1/3, 1/3;          % Center point
    0.6, 0.2;          % Vertex-weighted points
    0.2, 0.6;
    0.2, 0.2];
weights = [...
    -27/48;            % Center weight
    25/48;             % Vertex weights
    25/48;
    25/48];
end
