function [vertices, faces] = generate_sphere_mesh(subdivision_level)
    % Generate a spherical mesh via icosahedron subdivision
    % Input:  subdivision_level (e.g., 3 for ~1280 faces)
    % Output: vertices (Nx3), faces (Mx3)

    % Step 1: Create base icosahedron (12 vertices, 20 faces)
    [vertices, faces] = icosahedron();
    
    % Step 2: Subdivide faces and project onto sphere
    for i = 1:subdivision_level
        [vertices, faces] = subdivide_triangles(vertices, faces);
        vertices = vertices ./ vecnorm(vertices, 2, 2); % Project to unit sphere
    end
    
    % Optional: Plot the mesh
    plot_mesh(vertices, faces);
end

function [v, f] = icosahedron()
    % Return vertices and faces of an icosahedron
    t = (1 + sqrt(5)) / 2;
    v = [...
        -1,  t,  0;  1,  t,  0;  -1, -t,  0;  1, -t,  0; ...
         0, -1,  t;  0,  1,  t;  0, -1, -t;  0,  1, -t; ...
         t,  0, -1;  t,  0,  1; -t,  0, -1; -t,  0,  1];
    f = [...
         1, 12,  6;  1,  6,  2;  1,  2,  8;  1,  8, 11;  1, 11, 12; ...
         2,  6, 10;  6, 12,  5; 12, 11,  3; 11,  8,  7;  8,  2,  9; ...
         4, 10,  5;  4,  5,  3;  4,  3,  7;  4,  7,  9;  4,  9, 10; ...
         5, 10,  6;  3,  5, 12;  7,  3, 11;  9,  7,  8; 10,  9,  2];
end

function [v_new, f_new] = subdivide_triangles(v, f)
    % Subdivide each triangle into 4 smaller triangles
    nv = size(v, 1);
    nf = size(f, 1);
    v_new = v;
    f_new = zeros(nf * 4, 3);
    
    for i = 1:nf
        % Triangle vertices
        tri = f(i, :);
        v1 = v(tri(1), :);
        v2 = v(tri(2), :);
        v3 = v(tri(3), :);
        
        % Edge midpoints (new vertices)
        m12 = (v1 + v2) / 2;
        m23 = (v2 + v3) / 2;
        m31 = (v3 + v1) / 2;
        
        % Add new vertices
        m12_idx = size(v_new, 1) + 1;
        m23_idx = m12_idx + 1;
        m31_idx = m23_idx + 1;
        v_new = [v_new; m12; m23; m31];
        
        % New faces (4 per original triangle)
        f_new(4*(i-1)+1, :) = [tri(1), m12_idx, m31_idx];
        f_new(4*(i-1)+2, :) = [tri(2), m23_idx, m12_idx];
        f_new(4*(i-1)+3, :) = [tri(3), m31_idx, m23_idx];
        f_new(4*(i-1)+4, :) = [m12_idx, m23_idx, m31_idx];
    end
end

function plot_mesh(v, f)
    % Plot the mesh
    figure;
    trisurf(f, v(:, 1), v(:, 2), v(:, 3), 'FaceColor', 'cyan', 'EdgeColor', 'black');
    axis equal;
    title('Triangular Mesh on Sphere');
    xlabel('X'); ylabel('Y'); zlabel('Z');
end