import numpy as np
import plotly.graph_objects as go
import string

# 3D plotting helper
def plot_mse_3D(w1_vec, w2_vec, mse):
    # Obtain the index of the best weights
    w_index = np.array(np.unravel_index(mse.argmin(), mse.shape))

    fig = go.Figure(
        data=[
            go.Surface(z=mse, x=w1_vec, y=w2_vec, opacity=0.9),
            go.Scatter3d(x=[w1_vec[w_index[0]]], y=[w2_vec[w_index[1]]], 
                         z=[mse.min()])
        ]
    )
    fig.update_layout(title='Mean Squared Error Surface', height=600, width=800, 
                      autosize=False,
                      scene=dict(xaxis_title="w1", yaxis_title="w2", 
                                 zaxis_title="MSE")
    )
    fig.show()

    
# ===== Code checker for exercises =============================================
# 06a, Task 1
def check_bmi_function(f):
    n_tests = 1000
    heights = np.random.randint(150, 210, n_tests)
    weights = 70 + np.random.rand(n_tests) * 20
    
    def bmi_category(height_in_cm, weight_in_kg):
        bmi = weight_in_kg / (height_in_cm / 100) ** 2
        if bmi < 18.5:
            cat = "Underweight"
        elif bmi < 25:
            cat = "Normal"
        elif bmi < 30:
            cat = "Overweight"
        else:
            cat = "Obese"
        return cat
    
    if all([f(h, w) == bmi_category(h, w) for (h, w) in zip(heights, weights)]):
        print("✅ Your function works perfectly!")
    else:
        print("⛔ There is an error in your function.")

# 06b, Task 2
def check_town_canton_extractor(f):
    n_tests = 1000
    abc = np.array(list(string.ascii_lowercase))
    strings = [
        "".join(np.random.choice(abc, np.random.randint(1, 20)))
        + " " + "(" + "".join(np.random.choice(abc, np.random.randint(1, 20))) 
        + ")" for _ in range(n_tests)]

    def extract_town_canton(input_string):
        # Use .split to separate the town and canton
        town, canton = input_string.split(" ")
        canton = canton.strip("()")
        return town, canton # Output results
    
    if all([f(s) == extract_town_canton(s) for s in strings]):
        print("✅ Your function works perfectly!")
    else:
        print("⛔ There is an error in your function.")
    
# 06c, Task 1
def check_bin_returns(f):
    n_tests = 1000
    returns = np.random.rand(n_tests) * 20
    
    def bin_returns(x):
        # Create the string for the sign (positive/negative)
        sgn = "positive " if x > 0 else "negative "
        if abs(x) > 5: # Extreme returns
            adj = "extreme "
        elif abs(x) > 2: # Large returns
            adj = "large "
        else: # 'Normal' returns, adjective is blank
            adj = ""
        # Return the classification of returns
        return adj + sgn + "returns"
    
    if all([f(r) == bin_returns(r) for r in returns]):
        print("✅ Your function works perfectly!")
    else:
        print("⛔ There is an error in your function.")
        
# 113 SVD, Task 1
def check_truncated_svd(f):
    n_tests = 1000
    matrices = [np.random.rand(1 + np.random.randint(100), 
                               1 + np.random.randint(100)) for _ in range(n_tests)]
    ranks = [1 + np.random.randint(np.linalg.matrix_rank(A)) for A in matrices]
    
    
    def truncated_svd(A, k):
        # Perform SVD
        U, S, V = svd(A, full_matrices=False)
        # Truncate and return the matrices
        return U[:, :k], np.diag(S[:k]), V[:k, :]
    
    if all([all([np.all(x == y) for x, y in zip(f(A, k), truncated_svd(A, k))]) for (A, k) in zip(matrices, ranks)]):
        print("✅ Your function works perfectly!")
    else:
        print("⛔ There is an error in your function.")
    