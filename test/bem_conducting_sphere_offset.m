function bem_conducting_sphere_offset()
% Step 1: Generate mesh (icosphere, subdivision_level=3)
subdivision_level = 3;
[vertices, faces] = generate_sphere_mesh(subdivision_level);
num_vertices = size(vertices, 1);
num_faces = size(faces, 1);

% Step 2: Move observation points slightly outward (R = 1.01)
R_offset = 1.01;
% vertices_obs = R_offset * vertices;  % Observation points at R=1.01

% Step 3: Assemble BEM system (Ax = b)
A = zeros(num_faces, num_vertices);  % System matrix
b = ones(num_faces, 1);              % RHS (potential=1 at R=1.01)

% Quadrature rule for integration over triangles
[quad_points, quad_weights] = get_triangle_quadrature();

% Loop over observation points (at R=1.01)
for i = 1:num_faces
    % use the centroid of the triangular elements as the observation point
    tri = faces(i, :);
    r1 = vertices(tri(1), :);
    r2 = vertices(tri(2), :);
    r3 = vertices(tri(3), :);
    r_i = (r1 + r2 + r3)/3*R_offset;
    
       % Face normal at observation point (outward normal)
    face_normal = cross(r2 - r1, r3 - r1);
    n_hat = face_normal / norm(face_normal);

    % Build RHS: E ¡¤ n = 1 on upper, 0 on lower
    if r_i(3) > 0
        b(i) = 1;
    else
        b(i) = 0;
    end
    
    % Loop over source triangles
    % Loop over source triangles
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
            
            r_vec = r_i - r_src;
            r_norm = norm(r_vec);
            G_grad_dot_n = dot(r_vec, n_hat) / (4 * pi * r_norm^3);

            % Linear shape functions
            N1 = 1 - xi - eta;
            N2 = xi;
            N3 = eta;

            % Distribute to source vertices
            A(i, tri(1)) = A(i, tri(1)) + quad_weights(k) * G_grad_dot_n * J * N1;
            A(i, tri(2)) = A(i, tri(2)) + quad_weights(k) * G_grad_dot_n * J * N2;
            A(i, tri(3)) = A(i, tri(3)) + quad_weights(k) * G_grad_dot_n * J * N3;
        end
    end
end

% Step 4: Solve for charge density (sigma = A \ b)
 sigma = A \ b;
% sigma=lsqminnorm(A, b);

% Step 5: Visualize charge density
figure;
trisurf(faces, vertices(:, 1), vertices(:, 2), vertices(:, 3), sigma, 'EdgeColor', 'none');
axis equal; colorbar; title('Surface Charge Density \sigma (Offset R=1.01)');
xlabel('X'); ylabel('Y'); zlabel('Z');

% Step 6: Compare with analytical solution (sigma_true = epsilon0 for V=1)
sigma_true = 1;  % Assuming epsilon0 = 1 for simplicity
fprintf('Mean sigma: %.4f (Error: %.2f%%)\n', mean(sigma), 100*abs(mean(sigma)-sigma_true)/sigma_true);
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

