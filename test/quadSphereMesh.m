function [vertices, faces] = quadSphereMesh(n)
    % Generate sphere vertices (with overlapping poles)
    [X, Y, Z] = sphere(n);  
    vertices = [X(:), Y(:), Z(:)];
    [m, ~] = size(X);
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
end