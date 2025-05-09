# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nWjfEqyHAkRwgSpsSxjKjpVCUMXDflFU
"""

import numpy as np
import matplotlib.pyplot as plt

def stokes_flow_sphere_BEM():
    # Parameters
    R = 1.0  # Radius of the sphere
    U0 = np.array([0, 0, 1])  # Constant velocity of the sphere [Ux, Uy, Uz]
    N = 1000  # Number of surface elements (increased for accuracy)
    mu = 1.0  # Dynamic viscosity of the fluid

    # Discretize the sphere surface
    vertices, faces = generateSphereMesh(R, N)

    # Plot the triangular mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, alpha=0.5, edgecolor='k')
    ax.set_title('Sphere Triangular Mesh')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    plt.show()

    numNodes = vertices.shape[0]

    # Initialize the system matrix and right-hand side vector
    A = np.zeros((3 * numNodes, 3 * numNodes))
    B = np.zeros((3 * numNodes,))

    # Precompute triangle quadrature points and weights
    quad_points, quad_weights = get_triangle_quadrature()
    num_quads = len(quad_weights)



    # Loop over triangle faces (observation domain)
    for tri in range(faces.shape[0]):
        nodes = faces[tri, :]
        v1 = vertices[nodes[0], :]
        v2 = vertices[nodes[1], :]
        v3 = vertices[nodes[2], :]

        area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

        # Loop over each quadrature point
        for q in range(num_quads):
            a = quad_points[q, 0]
            b = quad_points[q, 1]
            c = 1 - a - b
            obs_point = a * v1 + b * v2 + c * v3

            shape = np.array([c, a, b])

            # Evaluate contribution of all nodal forces to this quadrature point
            for src_node in range(numNodes):
                src_point = vertices[src_node, :]
                r_vec = obs_point - src_point
                r = np.linalg.norm(r_vec)

                if r < 1e-12:
                    continue  # avoid singularity

                G = (np.eye(3)/r + np.outer(r_vec, r_vec)/r**3) / (8*np.pi*mu)

                for k in range(3):
                    row_idx = slice(3*(nodes[k]), 3*(nodes[k])+3)
                    col_idx = slice(3*(src_node), 3*(src_node)+3)

                    A[row_idx, col_idx] += shape[k] * G * quad_weights[q] * area

            # Distribute RHS velocity to triangle nodes
            for k in range(3):
                idx = slice(3*(nodes[k]), 3*(nodes[k])+3)
                B[idx] += shape[k] * U0 * quad_weights[q] * area



    # Solve for the force distribution
    f = np.linalg.solve(A, B)

    # Reshape the force distribution into (numNodes, 3)
    f = f.reshape((numNodes, 3))

    # Compute the total force
    total_force = np.sum(f, axis=0)

    # Analytical solution (Stokes' law)
    analytical_force = 6 * np.pi * mu * R * U0

    # Display results
    print(f'Computed total force: {total_force}')
    print(f'Analytical total force: {analytical_force}')

    # Visualize the force distribution (z-component)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], c=f[:,2], cmap='viridis')
    fig.colorbar(p)
    ax.set_title('Force Distribution on Sphere (z-component)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    plt.show()





def generateSphereMesh(R, N, tol=1e-10):
    """ Generate a spherical mesh with cleaned-up duplicate vertices. """

    m = int(np.round(np.sqrt(N / 2))) + 1
    u = np.linspace(0, np.pi, m)
    v = np.linspace(0, 2 * np.pi, m)

    u[-1] = np.pi  # ensure inclusion of the pole
    v[-1] = 2 * np.pi  # ensure closure in azimuth

    u, v = np.meshgrid(u, v)

    x = R * np.sin(u) * np.cos(v)
    y = R * np.sin(u) * np.sin(v)
    z = R * np.cos(u)

    # Flatten and round coordinates to remove numerical noise
    x = np.where(np.abs(x) < tol, 0.0, x)
    y = np.where(np.abs(y) < tol, 0.0, y)
    z = np.where(np.abs(z) < tol, 0.0, z)

    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    faces = []
    for i in range(m - 1):
        for j in range(m - 1):
            v1 = i * m + j
            v2 = i * m + j + 1
            v3 = (i + 1) * m + j + 1
            v4 = (i + 1) * m + j
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
    faces = np.array(faces)

    # Deduplicate vertices and update face indices
    vertices_rounded = np.round(vertices / tol) * tol  # bin by resolution
    _, idx_unique, idx_inverse = np.unique(vertices_rounded, axis=0, return_index=True, return_inverse=True)
    vertices = vertices[idx_unique]
    faces = idx_inverse[faces]

    return vertices, faces

def get_triangle_quadrature():
    """ 4-point Gaussian quadrature for triangles """
    points = np.array([
        [1/3, 1/3],
        [0.6, 0.2],
        [0.2, 0.6],
        [0.2, 0.2]
    ])
    weights = np.array([
        -27/48,
        25/48,
        25/48,
        25/48
    ])
    return points, weights

if __name__ == '__main__':
    stokes_flow_sphere_BEM()

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

cuda_kernel = r'''
extern "C" __global__
void bem_kernel(const double* vertices,
                const int* faces,
                const double* quad_points,
                const double* quad_weights,
                const double* U0,
                double* A,
                double* B,
                int num_faces,
                int num_nodes,
                int num_quads,
                double mu) {
    const float M_PI = 3.14159265358979323846f;

    int tri = blockDim.x * blockIdx.x + threadIdx.x;
    if (tri >= num_faces) return;

    // Each triangle has 3 nodes
    int n1 = faces[3 * tri + 0];
    int n2 = faces[3 * tri + 1];
    int n3 = faces[3 * tri + 2];

    double v1[3], v2[3], v3[3];
    for (int i = 0; i < 3; ++i) {
        v1[i] = vertices[3 * n1 + i];
        v2[i] = vertices[3 * n2 + i];
        v3[i] = vertices[3 * n3 + i];
    }

    // Compute area of the triangle
    double e1[3], e2[3], cross[3];
    for (int i = 0; i < 3; ++i) {
        e1[i] = v2[i] - v1[i];
        e2[i] = v3[i] - v1[i];
    }
    cross[0] = e1[1]*e2[2] - e1[2]*e2[1];
    cross[1] = e1[2]*e2[0] - e1[0]*e2[2];
    cross[2] = e1[0]*e2[1] - e1[1]*e2[0];
    double area = 0.5 * sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);

    for (int q = 0; q < num_quads; ++q) {
        double a = quad_points[2*q + 0];
        double b = quad_points[2*q + 1];
        double c = 1.0 - a - b;
        double shape[3] = {c, a, b};

        double obs[3] = {
            a*v1[0] + b*v2[0] + c*v3[0],
            a*v1[1] + b*v2[1] + c*v3[1],
            a*v1[2] + b*v2[2] + c*v3[2]
        };

        for (int src_node = 0; src_node < num_nodes; ++src_node) {
            double src[3] = {
                vertices[3 * src_node + 0],
                vertices[3 * src_node + 1],
                vertices[3 * src_node + 2]
            };

            double r_vec[3] = {
                obs[0] - src[0],
                obs[1] - src[1],
                obs[2] - src[2]
            };
            double r2 = r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2];
            if (r2 < 1e-24) continue;

            double r = sqrt(r2);
            double r3 = r2 * r;

            // Compute Stokeslet G
            double G[3][3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    G[i][j] = (i == j)/r + r_vec[i]*r_vec[j]/r3;

            double factor = quad_weights[q] * area / (8.0 * M_PI * mu);
            for (int k = 0; k < 3; ++k) {
                int row = 3 * faces[3*tri + k];
                int col = 3 * src_node;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        atomicAdd(&A[(row+i)*3*num_nodes + (col+j)],
                                  shape[k] * G[i][j] * factor);
            }
        }

        // Add contribution to RHS
        for (int k = 0; k < 3; ++k) {
            int idx = 3 * faces[3*tri + k];
            for (int i = 0; i < 3; ++i)
                atomicAdd(&B[idx+i], shape[k] * U0[i] * quad_weights[q] * area);
        }
    }
}
'''



def stokes_flow_sphere_BEM():
    # Parameters
    R = 1.0  # Radius of the sphere
    U0 = np.array([0, 0, 1])  # Constant velocity of the sphere [Ux, Uy, Uz]
    N = 1000  # Number of surface elements (increased for accuracy)
    mu = 1.0  # Dynamic viscosity of the fluid

    # Discretize the sphere surface
    vertices, faces = generateSphereMesh(R, N)

    # Plot the triangular mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, alpha=0.5, edgecolor='k')
    ax.set_title('Sphere Triangular Mesh')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    plt.show()

    numNodes = vertices.shape[0]

    # Initialize the system matrix and right-hand side vector
    A = np.zeros((3 * numNodes, 3 * numNodes))
    B = np.zeros((3 * numNodes,))

    # Precompute triangle quadrature points and weights
    quad_points, quad_weights = get_triangle_quadrature()
    num_quads = len(quad_weights)



   # Transfer all data to GPU
    vertices_gpu = cp.asarray(vertices, dtype=cp.float64).ravel()
    faces_gpu = cp.asarray(faces, dtype=cp.int32).ravel()
    quad_points_gpu = cp.asarray(quad_points, dtype=cp.float64).ravel()
    quad_weights_gpu = cp.asarray(quad_weights, dtype=cp.float64)
    U0_gpu = cp.asarray(U0, dtype=cp.float64)
    A_gpu = cp.zeros((3 * numNodes, 3 * numNodes), dtype=cp.float64)
    B_gpu = cp.zeros((3 * numNodes,), dtype=cp.float64)

# Compile and launch kernel
    module = cp.RawModule(code=cuda_kernel)
    kernel = module.get_function('bem_kernel')

    threads_per_block = 128
    num_blocks = (faces.shape[0] + threads_per_block - 1) // threads_per_block

    kernel((num_blocks,), (threads_per_block,),
       (vertices_gpu, faces_gpu, quad_points_gpu, quad_weights_gpu,
        U0_gpu, A_gpu, B_gpu,
        faces.shape[0], vertices.shape[0], quad_points.shape[0], mu))


  # Solve for the force distribution


    f = cp.linalg.solve(A_gpu, B_gpu)

    # Reshape the force distribution into (numNodes, 3)
    f = f.reshape((numNodes, 3))

    # Compute the total force
    total_force = cp.sum(f, axis=0)
    total_force = total_force.get()
    f=f.get()

    # Analytical solution (Stokes' law)
    analytical_force = 6 * np.pi * mu * R * U0

    # Display results
    print(f'Computed total force: {total_force}')
    print(f'Analytical total force: {analytical_force}')

    # Visualize the force distribution (z-component)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], c=f[:,2], cmap='viridis')
    fig.colorbar(p)
    ax.set_title('Force Distribution on Sphere (z-component)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    plt.show()





def generateSphereMesh(R, N, tol=1e-10):
    """ Generate a spherical mesh with cleaned-up duplicate vertices. """

    m = int(np.round(np.sqrt(N / 2))) + 1
    u = np.linspace(0, np.pi, m)
    v = np.linspace(0, 2 * np.pi, m)

    u[-1] = np.pi  # ensure inclusion of the pole
    v[-1] = 2 * np.pi  # ensure closure in azimuth

    u, v = np.meshgrid(u, v)

    x = R * np.sin(u) * np.cos(v)
    y = R * np.sin(u) * np.sin(v)
    z = R * np.cos(u)

    # Flatten and round coordinates to remove numerical noise
    x = np.where(np.abs(x) < tol, 0.0, x)
    y = np.where(np.abs(y) < tol, 0.0, y)
    z = np.where(np.abs(z) < tol, 0.0, z)

    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    faces = []
    for i in range(m - 1):
        for j in range(m - 1):
            v1 = i * m + j
            v2 = i * m + j + 1
            v3 = (i + 1) * m + j + 1
            v4 = (i + 1) * m + j
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
    faces = np.array(faces)

    # Deduplicate vertices and update face indices
    vertices_rounded = np.round(vertices / tol) * tol  # bin by resolution
    _, idx_unique, idx_inverse = np.unique(vertices_rounded, axis=0, return_index=True, return_inverse=True)
    vertices = vertices[idx_unique]
    faces = idx_inverse[faces]

    return vertices, faces

def get_triangle_quadrature():
    """ 4-point Gaussian quadrature for triangles """
    points = np.array([
        [1/3, 1/3],
        [0.6, 0.2],
        [0.2, 0.6],
        [0.2, 0.2]
    ])
    weights = np.array([
        -27/48,
        25/48,
        25/48,
        25/48
    ])
    return points, weights

if __name__ == '__main__':
    stokes_flow_sphere_BEM()

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# CUDA kernel code
kernel_code = r'''
extern "C" __global__
void diffusion_step(float* u, float* u_new, float D, float dx, float dt, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx > 0 && idx < n-1) { // avoid boundaries
        float laplacian = (u[idx-1] - 2.0f * u[idx] + u[idx+1]) / (dx * dx);
        u_new[idx] = u[idx] + D * dt * laplacian;
    }
}
'''

# Compile the kernel
diffusion_step = cp.RawKernel(kernel_code, 'diffusion_step')

# Problem parameters
n_points = 256
D = 0.1           # diffusion coefficient
dx = 1.0 / n_points
dt = 0.25 * dx * dx / D  # Stability condition
steps = 500       # number of time steps

# Initial condition: a spike in the middle
u = cp.zeros(n_points, dtype=cp.float32)
u[n_points//2] = 1.0
u_new = cp.zeros_like(u)

# Prepare CUDA grid
threads_per_block = 128
blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block

# Time-stepping loop
for step in range(steps):
    diffusion_step((blocks_per_grid,), (threads_per_block,), (u, u_new, D, dx, dt, n_points))
    u, u_new = u_new, u  # swap

# Download result to CPU and plot
u_cpu = cp.asnumpy(u)

x = np.linspace(0, 1, n_points)
plt.plot(x, u_cpu)
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Diffusion after {} steps'.format(steps))
plt.show()