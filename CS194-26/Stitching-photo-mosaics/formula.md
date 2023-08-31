# Homography Matrix

$$ p' = Hp $$

$$\begin{bmatrix}
wx'\\
wy'\\
w
\end{bmatrix} = 
\begin{bmatrix}
a_{11}&a_{12}&a_{13}\\
a_{21}&a_{22}&a_{23}\\
a_{31}&a_{32}&a_{33}
\end{bmatrix}
\begin{bmatrix}
x\\
y\\
1
\end{bmatrix}$$

$a_{33}$ is a scaling factor and can be set to 1. 

$$\begin{bmatrix}
wx'\\
wy'\\
w
\end{bmatrix} = 
\begin{bmatrix}
a_{11}&a_{12}&a_{13}\\
a_{21}&a_{22}&a_{23}\\
a_{31}&a_{32}&1
\end{bmatrix}
\begin{bmatrix}
x\\
y\\
1
\end{bmatrix}
$$

Therefore, we only have to solve 8 unknowns.

Set up **a system of linear equations** : $Ah=b$ , where 

$$h = \begin{bmatrix} 
a_{11} \\
a_{12} \\
a_{13} \\
a_{21} \\
a_{22} \\
a_{23} \\
a_{31} \\
a_{32} 
\end{bmatrix}$$

Before solving $Ah=b$, we have to define matrix $A$ and $b$ .

Expand the matrix multiplication. We get three equations:

$$\begin{cases}
a_{11}x + a_{12}y + a_{13} = wx' \\
a_{21}x + a_{22}y + a_{23} = wy'  \\
a_{31}x + a_{32}y + 1 = w 
\end{cases}$$

Both sides of the first and second equations are simultaneously multiplied by both sides of the third equation:

$$a_{11}x + a_{12}y + a_{13} = x'(a_{31}x + a_{32}y + 1)$$

$$a_{21}x + a_{22}y + a_{23} = y'(a_{31}x + a_{32}y + 1)$$

Then, the equations can be written as follow:

$$a_{11}x + a_{12}y + a_{13} - x'(a_{31}x + a_{32}y + 1) = 0 \Longrightarrow a_{11}x + a_{12}y + a_{13} - a_{31}xx' - a_{32}x'y = x'$$

$$a_{21}x + a_{22}y + a_{23} - y'(a_{31}x + a_{32}y + 1) = 0 \Longrightarrow a_{21}x + a_{22}y + a_{23} - a_{31}xy' - a_{32}yy' = y'$$

, and we represent the equations in matrix form:

$$\begin{bmatrix}
x&y&1&0&0&0&-xx'&-x'y \\
0&0&0&x&y&1&-xy'&-yy'
\end{bmatrix}
\begin{bmatrix}
a_{11} \\ 
a_{12} \\ 
a_{13} \\ 
a_{21} \\ 
a_{22} \\ 
a_{23} \\ 
a_{31} \\ 
a_{32}
\end{bmatrix} = 
\begin{bmatrix}
x' \\ 
y'
\end{bmatrix}
$$

There are 8 unknowns, we need at least 8 equations, that is, we need at least 4 points to solve $Ah=b$ .
To solve $Ah=b$, we use least-squares:

$$min \begin{Vmatrix} Ah-b \end{Vmatrix}^2$$
