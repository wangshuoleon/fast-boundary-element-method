function  [vertices, faces, sigma]=bem_conducting_sphere_offset_2()
    % Step 1: Generate mesh (icosphere, subdivision_level=3)
    close all
    subdivision_level = 2;
    [vertices, faces] = generate_sphere_mesh(subdivision_level);
    num_vertices = size(vertices, 1);
    num_faces = size(faces, 1);
    offset=1.1;

    % Step 2: Assemble BEM system (Ax = b)
    A = zeros(num_faces, num_vertices);  % System matrix
    b = ones(num_faces, 1);              % RHS (potential=1 at R=1.01)

    % Quadrature rule for integration over triangles
    [quad_points, quad_weights] = get_triangle_quadrature();

    % Loop over observation points (face centroids)
    for i = 1:num_faces
        tri = faces(i, :);
        r1 = vertices(tri(1), :);
        r2 = vertices(tri(2), :);
        r3 = vertices(tri(3), :);
        r_i = (r1 + r2 + r3)/3*offset;  % Centroid with offset
        
        % Face normal
        face_normal = cross(r2 - r1, r3 - r1);
        n_hat = face_normal / norm(face_normal);
        
        % Build RHS
        b(i) = (r_i(3) > 0);  % 1 on upper hemisphere, 0 on lower
        
        % Loop over source triangles
        for j = 1:num_faces
            tri_src = faces(j, :);
            r1_src = vertices(tri_src(1), :);
            r2_src = vertices(tri_src(2), :);
            r3_src = vertices(tri_src(3), :);
            
            edge1 = r2_src - r1_src;
            edge2 = r3_src - r1_src;
            J = norm(cross(edge1, edge2)) / 2;

            for k = 1:size(quad_points, 1)
                xi = quad_points(k, 1);
                eta = quad_points(k, 2);
                r_src = (1 - xi - eta) * r1_src + xi * r2_src + eta * r3_src;
                
                r_vec = r_i - r_src;
                r_norm = norm(r_vec);
                G_grad_dot_n = dot(r_vec, n_hat) / (4 * pi * r_norm^3);

                % Linear shape functions
                N1 = 1 - xi - eta;
                N2 = xi;
                N3 = eta;

                % Distribute to source vertices
                A(i, tri_src(1)) = A(i, tri_src(1)) + quad_weights(k) * G_grad_dot_n * J * N1;
                A(i, tri_src(2)) = A(i, tri_src(2)) + quad_weights(k) * G_grad_dot_n * J * N2;
                A(i, tri_src(3)) = A(i, tri_src(3)) + quad_weights(k) * G_grad_dot_n * J * N3;
            end
        end
    end

    % Step 3: Solve for charge density
    sigma = A \ b;
    
    % Step 4: Generate observation points using sphere function
    [X, Y, Z] = sphere(50);  % 50x50 grid
    obs_points = [X(:), Y(:), Z(:)]*offset*1.1;
    num_obs = size(obs_points, 1);
    
    % Step 5: Compute potential at observation points
    potential = zeros(num_obs, 1);
    for i = 1:num_obs
        r_obs = obs_points(i, :);
        for j = 1:num_faces
            tri = faces(j, :);
            r1 = vertices(tri(1), :);
            r2 = vertices(tri(2), :);
            r3 = vertices(tri(3), :);
            
            edge1 = r2 - r1;
            edge2 = r3 - r1;
            J = norm(cross(edge1, edge2)) / 2;
            
            for k = 1:size(quad_points, 1)
                xi = quad_points(k, 1);
                eta = quad_points(k, 2);
                r_src = (1 - xi - eta) * r1 + xi * r2 + eta * r3;
                
                G = 1 / (4 * pi * norm(r_obs - r_src));
                
                % Linear shape functions
                N1 = 1 - xi - eta;
                N2 = xi;
                N3 = eta;
                
                % Contribution from this quadrature point
                contrib = quad_weights(k) * G * J;
                potential(i) = potential(i) + contrib * (sigma(tri(1))*N1 + sigma(tri(2))*N2 + sigma(tri(3))*N3);
            end
        end
    end
    
    % Step 6: Visualize results
    figure;
    
    % Subplot 1: Charge density on original mesh
    subplot(1,2,1);
    trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3), sigma, 'EdgeColor', 'none');
    axis equal; colorbar; title('Surface Charge Density \sigma');
    
    % Subplot 2: Potential on observation sphere
    subplot(1,2,2);
    potential_grid = reshape(potential, size(X));
    surf(X, Y, Z, potential_grid, 'EdgeColor', 'none');
    axis equal; colorbar; title('Computed Potential on Sphere');
    % caxis([0 1]);  % Fix color range for comparison
    
    fprintf('Mean charge density: %.4f\n', mean(sigma));
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