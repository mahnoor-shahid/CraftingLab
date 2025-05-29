import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="MLE vs EM Algorithm Demo",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .concept-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .algorithm-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    .comparison-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎯 Maximum Likelihood vs Expectation Maximization</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="concept-box">
    <h3>🧠 What You'll Learn:</h3>
    <ul>
        <li><strong>Maximum Likelihood Estimation (MLE):</strong> Finding parameters that make observed data most likely</li>
        <li><strong>Expectation Maximization (EM):</strong> Iterative algorithm for MLE with hidden/missing data</li>
        <li><strong>Key Difference:</strong> MLE works with complete data, EM works with incomplete/hidden data</li>
        <li><strong>Applications:</strong> Gaussian mixtures, clustering, missing data problems</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🧭 Choose Demo Type")
demo_type = st.sidebar.selectbox(
    "Select what to explore:",
    [
        "📊 Basic Concepts Comparison",
        "🎯 Simple MLE Example", 
        "🔄 EM Algorithm Visualization",
        "📈 Gaussian Mixture Models",
        "🧩 Missing Data Problem",
        "⚖️ Side-by-Side Comparison"
    ]
)

def generate_gaussian_data(n=100, mu=0, sigma=1):
    """Generate data from normal distribution"""
    return np.random.normal(mu, sigma, n)

def gaussian_pdf(x, mu, sigma):
    """Gaussian probability density function"""
    return (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-0.5*((x-mu)/sigma)**2)

def log_likelihood_gaussian(params, data):
    """Log likelihood for Gaussian distribution"""
    mu, sigma = params
    if sigma <= 0:
        return -np.inf
    ll = np.sum(np.log(gaussian_pdf(data, mu, sigma)))
    return -ll  # Negative for minimization

def mle_gaussian(data):
    """Maximum Likelihood Estimation for Gaussian"""
    # Analytical solution for Gaussian
    mu_mle = np.mean(data)
    sigma_mle = np.sqrt(np.mean((data - mu_mle)**2))
    return mu_mle, sigma_mle

class GaussianMixtureEM:
    """Simple 2-component Gaussian Mixture using EM"""
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.weights = None
        self.means = None
        self.variances = None
        self.history = []
        
    def initialize_parameters(self, data):
        """Initialize parameters randomly"""
        n = len(data)
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = np.random.choice(data, self.n_components)
        self.variances = np.ones(self.n_components)
        
    def e_step(self, data):
        """Expectation step: compute responsibilities"""
        n = len(data)
        responsibilities = np.zeros((n, self.n_components))
        
        for k in range(self.n_components):
            responsibilities[:, k] = (self.weights[k] * 
                                    gaussian_pdf(data, self.means[k], np.sqrt(self.variances[k])))
        
        # Normalize responsibilities
        row_sums = responsibilities.sum(axis=1, keepdims=True)
        responsibilities = responsibilities / (row_sums + 1e-10)
        
        return responsibilities
    
    def m_step(self, data, responsibilities):
        """Maximization step: update parameters"""
        n = len(data)
        
        # Update weights
        self.weights = responsibilities.mean(axis=0)
        
        # Update means
        for k in range(self.n_components):
            self.means[k] = np.sum(responsibilities[:, k] * data) / np.sum(responsibilities[:, k])
        
        # Update variances
        for k in range(self.n_components):
            diff = data - self.means[k]
            self.variances[k] = np.sum(responsibilities[:, k] * diff**2) / np.sum(responsibilities[:, k])
    
    def compute_log_likelihood(self, data):
        """Compute log likelihood of data"""
        likelihood = np.zeros(len(data))
        for k in range(self.n_components):
            likelihood += (self.weights[k] * 
                         gaussian_pdf(data, self.means[k], np.sqrt(self.variances[k])))
        return np.sum(np.log(likelihood + 1e-10))
    
    def fit(self, data, max_iters=50, tol=1e-6):
        """Fit the model using EM algorithm"""
        self.initialize_parameters(data)
        self.history = []
        
        prev_ll = -np.inf
        
        for i in range(max_iters):
            # E-step
            responsibilities = self.e_step(data)
            
            # M-step
            self.m_step(data, responsibilities)
            
            # Compute log likelihood
            ll = self.compute_log_likelihood(data)
            
            # Store history
            self.history.append({
                'iteration': i,
                'log_likelihood': ll,
                'means': self.means.copy(),
                'variances': self.variances.copy(),
                'weights': self.weights.copy()
            })
            
            # Check convergence
            if abs(ll - prev_ll) < tol:
                break
                
            prev_ll = ll
        
        return self

# Main content based on selection
if demo_type == "📊 Basic Concepts Comparison":
    st.markdown("## 📊 Basic Concepts: MLE vs EM")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="algorithm-box">
            <h4>🎯 Maximum Likelihood Estimation (MLE)</h4>
            <p><strong>Purpose:</strong> Find parameters θ that maximize P(data|θ)</p>
            <p><strong>When to use:</strong> Complete data, known model</p>
            <p><strong>How it works:</strong></p>
            <ol>
                <li>Define likelihood function L(θ|data)</li>
                <li>Take derivative and set to zero</li>
                <li>Solve for θ (analytical or numerical)</li>
            </ol>
            <p><strong>Example:</strong> Estimating mean/variance of normal distribution</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="algorithm-box">
            <h4>🔄 Expectation Maximization (EM)</h4>
            <p><strong>Purpose:</strong> MLE when there are hidden/missing variables</p>
            <p><strong>When to use:</strong> Incomplete data, latent variables</p>
            <p><strong>How it works:</strong></p>
            <ol>
                <li><strong>E-step:</strong> Estimate hidden variables given current params</li>
                <li><strong>M-step:</strong> Update parameters given estimated hidden vars</li>
                <li>Repeat until convergence</li>
            </ol>
            <p><strong>Example:</strong> Gaussian mixture models, clustering</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="comparison-box">
        <h4>⚖️ Key Differences:</h4>
        <table>
            <tr><th>Aspect</th><th>MLE</th><th>EM</th></tr>
            <tr><td><strong>Data Type</strong></td><td>Complete, observed</td><td>Incomplete, with hidden variables</td></tr>
            <tr><td><strong>Solution</strong></td><td>Often analytical</td><td>Iterative algorithm</td></tr>
            <tr><td><strong>Complexity</strong></td><td>Usually simpler</td><td>More complex</td></tr>
            <tr><td><strong>Convergence</strong></td><td>Direct</td><td>Guaranteed to local maximum</td></tr>
            <tr><td><strong>Applications</strong></td><td>Parameter estimation</td><td>Clustering, missing data</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

elif demo_type == "🎯 Simple MLE Example":
    st.markdown("## 🎯 Maximum Likelihood Estimation Demo")
    
    st.markdown("""
    <div class="concept-box">
        <h4>🎲 Scenario: Estimating Parameters of a Normal Distribution</h4>
        <p>You have some data points and want to find the mean (μ) and standard deviation (σ) 
        that make this data most likely to have occurred.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Data Generation Controls")
        
        true_mu = st.slider("True Mean (μ)", -5.0, 5.0, 2.0, 0.1)
        true_sigma = st.slider("True Standard Deviation (σ)", 0.5, 3.0, 1.0, 0.1)
        n_samples = st.slider("Number of Data Points", 20, 500, 100)
        
        # Generate data
        np.random.seed(42)
        data = generate_gaussian_data(n_samples, true_mu, true_sigma)
        
        # MLE estimation
        mu_mle, sigma_mle = mle_gaussian(data)
        
        st.markdown("### 📊 MLE Results")
        st.metric("Estimated Mean", f"{mu_mle:.3f}", f"{mu_mle - true_mu:.3f}")
        st.metric("Estimated Std Dev", f"{sigma_mle:.3f}", f"{sigma_mle - true_sigma:.3f}")
        st.metric("Sample Size", n_samples)
    
    with col2:
        # Plot data and fitted distribution
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=['Data Histogram with MLE Fit', 'Likelihood Surface'])
        
        # Histogram with fitted curve
        fig.add_trace(
            go.Histogram(x=data, nbinsx=20, name='Data', 
                        histnorm='probability density', opacity=0.7),
            row=1, col=1
        )
        
        # True distribution
        x_range = np.linspace(data.min() - 1, data.max() + 1, 200)
        true_pdf = gaussian_pdf(x_range, true_mu, true_sigma)
        fig.add_trace(
            go.Scatter(x=x_range, y=true_pdf, mode='lines', 
                      name='True Distribution', line=dict(color='red', width=3)),
            row=1, col=1
        )
        
        # MLE fitted distribution
        mle_pdf = gaussian_pdf(x_range, mu_mle, sigma_mle)
        fig.add_trace(
            go.Scatter(x=x_range, y=mle_pdf, mode='lines', 
                      name='MLE Fit', line=dict(color='blue', width=3, dash='dash')),
            row=1, col=1
        )
        
        # Likelihood surface
        mu_range = np.linspace(true_mu - 2, true_mu + 2, 50)
        sigma_range = np.linspace(0.5, 3.0, 50)
        
        likelihood_surface = np.zeros((len(sigma_range), len(mu_range)))
        for i, sigma in enumerate(sigma_range):
            for j, mu in enumerate(mu_range):
                likelihood_surface[i, j] = -log_likelihood_gaussian([mu, sigma], data)
        
        fig.add_trace(
            go.Contour(x=mu_range, y=sigma_range, z=likelihood_surface,
                      name='Log-Likelihood', colorscale='Viridis'),
            row=2, col=1
        )
        
        # Mark MLE point
        fig.add_trace(
            go.Scatter(x=[mu_mle], y=[sigma_mle], mode='markers',
                      marker=dict(color='red', size=10, symbol='star'),
                      name='MLE Solution'),
            row=2, col=1
        )
        
        fig.update_layout(height=700, title="MLE for Normal Distribution")
        fig.update_xaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_xaxes(title_text="Mean (μ)", row=2, col=1)
        fig.update_yaxes(title_text="Std Dev (σ)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Mathematical explanation
    st.markdown("""
    ### 🧮 Mathematical Details
    
    **Likelihood Function for Normal Distribution:**
    ```
    L(μ,σ|data) = ∏ᵢ (1/√(2πσ²)) × exp(-(xᵢ-μ)²/(2σ²))
    ```
    
    **Log-Likelihood (easier to work with):**
    ```
    ℓ(μ,σ) = -n/2 × ln(2πσ²) - Σᵢ(xᵢ-μ)²/(2σ²)
    ```
    
    **MLE Solutions (analytical):**
    - μ̂ = (1/n) × Σᵢxᵢ  (sample mean)
    - σ̂² = (1/n) × Σᵢ(xᵢ-μ̂)²  (sample variance)
    """)

elif demo_type == "🔄 EM Algorithm Visualization":
    st.markdown("## 🔄 EM Algorithm Step-by-Step")
    
    st.markdown("""
    <div class="concept-box">
        <h4>🎯 Watch EM Algorithm in Action</h4>
        <p>This demo shows how EM iteratively improves parameter estimates for a 2-component Gaussian mixture.</p>
        <p><strong>Key Insight:</strong> EM alternates between estimating hidden cluster assignments (E-step) 
        and updating parameters (M-step).</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Mixture Model Setup")
        
        # True mixture parameters
        mu1_true = st.slider("Component 1 Mean", -5.0, 5.0, -1.0, 0.1)
        mu2_true = st.slider("Component 2 Mean", -5.0, 5.0, 2.0, 0.1)
        sigma1_true = st.slider("Component 1 Std", 0.5, 2.0, 0.8, 0.1)
        sigma2_true = st.slider("Component 2 Std", 0.5, 2.0, 1.2, 0.1)
        weight1_true = st.slider("Component 1 Weight", 0.1, 0.9, 0.6, 0.1)
        
        n_samples = st.slider("Number of Samples", 100, 500, 200)
        max_iterations = st.slider("Max EM Iterations", 5, 50, 20)
        
        # Generate mixture data
        np.random.seed(42)
        n1 = int(n_samples * weight1_true)
        n2 = n_samples - n1
        
        data1 = np.random.normal(mu1_true, sigma1_true, n1)
        data2 = np.random.normal(mu2_true, sigma2_true, n2)
        data = np.concatenate([data1, data2])
        np.random.shuffle(data)
        
        # True cluster assignments (for visualization)
        true_labels = np.concatenate([np.zeros(n1), np.ones(n2)])
        np.random.shuffle(true_labels)
    
    with col2:
        # Run EM algorithm
        em_model = GaussianMixtureEM(n_components=2)
        em_model.fit(data, max_iters=max_iterations)
        
        # Animation controls
        iteration_to_show = st.slider("Show Iteration", 0, len(em_model.history)-1, len(em_model.history)-1)
        
        current_state = em_model.history[iteration_to_show]
        
        # Plot current state
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=[
                               f'Iteration {iteration_to_show}: Mixture Components',
                               'Log-Likelihood Progress',
                               'Parameter Evolution: Means',
                               'Parameter Evolution: Weights'
                           ])
        
        # Data histogram with current fit
        fig.add_trace(
            go.Histogram(x=data, nbinsx=30, name='Data', 
                        histnorm='probability density', opacity=0.5),
            row=1, col=1
        )
        
        # Current mixture components
        x_range = np.linspace(data.min()-1, data.max()+1, 200)
        
        for k in range(2):
            component_pdf = (current_state['weights'][k] * 
                           gaussian_pdf(x_range, current_state['means'][k], 
                                      np.sqrt(current_state['variances'][k])))
            fig.add_trace(
                go.Scatter(x=x_range, y=component_pdf, mode='lines',
                          name=f'Component {k+1}', 
                          line=dict(width=2)),
                row=1, col=1
            )
        
        # Total mixture
        total_pdf = np.zeros_like(x_range)
        for k in range(2):
            total_pdf += (current_state['weights'][k] * 
                         gaussian_pdf(x_range, current_state['means'][k], 
                                    np.sqrt(current_state['variances'][k])))
        
        fig.add_trace(
            go.Scatter(x=x_range, y=total_pdf, mode='lines',
                      name='Total Mixture', line=dict(color='black', width=3)),
            row=1, col=1
        )
        
        # Log-likelihood progress
        iterations = [h['iteration'] for h in em_model.history[:iteration_to_show+1]]
        log_likelihoods = [h['log_likelihood'] for h in em_model.history[:iteration_to_show+1]]
        
        fig.add_trace(
            go.Scatter(x=iterations, y=log_likelihoods, mode='lines+markers',
                      name='Log-Likelihood', line=dict(color='red')),
            row=1, col=2
        )
        
        # Parameter evolution - means
        means_1 = [h['means'][0] for h in em_model.history[:iteration_to_show+1]]
        means_2 = [h['means'][1] for h in em_model.history[:iteration_to_show+1]]
        
        fig.add_trace(
            go.Scatter(x=iterations, y=means_1, mode='lines+markers',
                      name='Mean 1', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=iterations, y=means_2, mode='lines+markers',
                      name='Mean 2', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Add true values as horizontal lines
        fig.add_hline(y=mu1_true, line_dash="dash", line_color="blue", 
                     annotation_text="True Mean 1", row=2, col=1)
        fig.add_hline(y=mu2_true, line_dash="dash", line_color="orange", 
                     annotation_text="True Mean 2", row=2, col=1)
        
        # Parameter evolution - weights
        weights_1 = [h['weights'][0] for h in em_model.history[:iteration_to_show+1]]
        weights_2 = [h['weights'][1] for h in em_model.history[:iteration_to_show+1]]
        
        fig.add_trace(
            go.Scatter(x=iterations, y=weights_1, mode='lines+markers',
                      name='Weight 1', line=dict(color='purple')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=iterations, y=weights_2, mode='lines+markers',
                      name='Weight 2', line=dict(color='green')),
            row=2, col=2
        )
        
        # Add true weights
        fig.add_hline(y=weight1_true, line_dash="dash", line_color="purple", 
                     annotation_text="True Weight 1", row=2, col=2)
        fig.add_hline(y=1-weight1_true, line_dash="dash", line_color="green", 
                     annotation_text="True Weight 2", row=2, col=2)
        
        fig.update_layout(height=800, title="EM Algorithm Convergence")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show current parameters
        st.markdown("### 📊 Current EM Estimates")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Component 1 Mean", 
                     f"{current_state['means'][0]:.3f}",
                     f"{current_state['means'][0] - mu1_true:.3f}")
            st.metric("Component 2 Mean", 
                     f"{current_state['means'][1]:.3f}",
                     f"{current_state['means'][1] - mu2_true:.3f}")
        
        with col_b:
            st.metric("Component 1 Weight", 
                     f"{current_state['weights'][0]:.3f}",
                     f"{current_state['weights'][0] - weight1_true:.3f}")
            st.metric("Component 2 Weight", 
                     f"{current_state['weights'][1]:.3f}",
                     f"{current_state['weights'][1] - (1-weight1_true):.3f}")
        
        with col_c:
            st.metric("Log-Likelihood", f"{current_state['log_likelihood']:.2f}")
            st.metric("Iteration", f"{iteration_to_show}")

elif demo_type == "📈 Gaussian Mixture Models":
    st.markdown("## 📈 Gaussian Mixture Models: The Classic EM Application")
    
    st.markdown("""
    <div class="concept-box">
        <h4>🎯 Why GMM Needs EM</h4>
        <p><strong>Problem:</strong> We have data from multiple Gaussian distributions mixed together, 
        but we don't know which data point came from which distribution.</p>
        <p><strong>Hidden Variables:</strong> The cluster assignments (which Gaussian each point belongs to)</p>
        <p><strong>EM Solution:</strong> Iteratively estimate cluster assignments and update parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive GMM demo
    st.markdown("### 🎮 Interactive GMM Demo")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🎛️ GMM Configuration")
        
        n_components = st.selectbox("Number of Components", [2, 3, 4], index=0)
        n_samples_per_component = st.slider("Samples per Component", 50, 200, 100)
        overlap_level = st.slider("Component Overlap", 0.5, 3.0, 1.5, 0.1)
        
        # Generate complex mixture data
        np.random.seed(42)
        
        if n_components == 2:
            centers = [(-2, 0), (2, 0)]
            colors = ['red', 'blue']
        elif n_components == 3:
            centers = [(-2, -1), (2, -1), (0, 2)]
            colors = ['red', 'blue', 'green']
        else:  # 4 components
            centers = [(-2, -2), (2, -2), (-2, 2), (2, 2)]
            colors = ['red', 'blue', 'green', 'purple']
        
        # Generate 2D mixture data
        all_data = []
        true_labels = []
        
        for i, (cx, cy) in enumerate(centers):
            component_data = np.random.multivariate_normal(
                [cx, cy], 
                [[overlap_level, 0], [0, overlap_level]], 
                n_samples_per_component
            )
            all_data.append(component_data)
            true_labels.extend([i] * n_samples_per_component)
        
        data_2d = np.vstack(all_data)
        true_labels = np.array(true_labels)
        
        # Shuffle data
        shuffle_idx = np.random.permutation(len(data_2d))
        data_2d = data_2d[shuffle_idx]
        true_labels = true_labels[shuffle_idx]
        
        show_true_clusters = st.checkbox("Show True Clusters", False)
        show_em_progress = st.checkbox("Show EM Progress", True)
    
    with col2:
        if show_em_progress:
            # Simple 2D EM implementation
            class GMM2D:
                def __init__(self, n_components):
                    self.n_components = n_components
                    self.history = []
                
                def fit(self, data, max_iters=20):
                    n, d = data.shape
                    
                    # Initialize parameters
                    self.weights = np.ones(self.n_components) / self.n_components
                    self.means = data[np.random.choice(n, self.n_components, replace=False)]
                    self.covariances = [np.eye(d) for _ in range(self.n_components)]
                    
                    self.history = []
                    
                    for iteration in range(max_iters):
                        # E-step: compute responsibilities
                        responsibilities = np.zeros((n, self.n_components))
                        
                        for k in range(self.n_components):
                            diff = data - self.means[k]
                            # Simplified: assume diagonal covariance
                            cov_det = np.linalg.det(self.covariances[k])
                            cov_inv = np.linalg.inv(self.covariances[k])
                            
                            mahalanobis = np.sum((diff @ cov_inv) * diff, axis=1)
                            responsibilities[:, k] = (self.weights[k] * 
                                                    np.exp(-0.5 * mahalanobis) / 
                                                    np.sqrt((2*np.pi)**d * cov_det))
                        
                        # Normalize responsibilities
                        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
                        
                        # M-step: update parameters
                        for k in range(self.n_components):
                            Nk = responsibilities[:, k].sum()
                            self.weights[k] = Nk / n
                            self.means[k] = (responsibilities[:, k:k+1] * data).sum(axis=0) / Nk
                            
                            diff = data - self.means[k]
                            self.covariances[k] = ((responsibilities[:, k:k+1] * diff).T @ diff) / Nk
                        
                        # Store history
                        self.history.append({
                            'iteration': iteration,
                            'means': self.means.copy(),
                            'weights': self.weights.copy(),
                            'responsibilities': responsibilities.copy()
                        })
            
            # Fit GMM
            gmm = GMM2D(n_components)
            gmm.fit(data_2d, max_iters=15)
            
            # Show iteration slider
            iteration_to_show = st.slider("EM Iteration", 0, len(gmm.history)-1, len(gmm.history)-1)
            current_gmm_state = gmm.history[iteration_to_show]
            
            # Plot results
            fig = go.Figure()
            
            if show_true_clusters:
                # Show true clusters
                for i in range(n_components):
                    mask = true_labels == i
                    fig.add_trace(go.Scatter(
                        x=data_2d[mask, 0], y=data_2d[mask, 1],
                        mode='markers', name=f'True Cluster {i+1}',
                        marker=dict(color=colors[i], size=6, opacity=0.7)
                    ))
            else:
                # Show EM clustering results
                responsibilities = current_gmm_state['responsibilities']
                predicted_labels = np.argmax(responsibilities, axis=1)
                
                for i in range(n_components):
                    mask = predicted_labels == i
                    fig.add_trace(go.Scatter(
                        x=data_2d[mask, 0], y=data_2d[mask, 1],
                        mode='markers', name=f'EM Cluster {i+1}',
                        marker=dict(color=colors[i], size=6, opacity=0.7)
                    ))
            
            # Show current means
            current_means = current_gmm_state['means']
            fig.add_trace(go.Scatter(
                x=current_means[:, 0], y=current_means[:, 1],
                mode='markers', name='Estimated Centers',
                marker=dict(color='black', size=15, symbol='x', line=dict(width=3))
            ))
            
            # Show true centers
            true_centers = np.array(centers)
            fig.add_trace(go.Scatter(
                x=true_centers[:, 0], y=true_centers[:, 1],
                mode='markers', name='True Centers',
                marker=dict(color='white', size=12, symbol='star', 
                           line=dict(color='black', width=2))
            ))
            
            fig.update_layout(title=f"2D Gaussian Mixture Model - Iteration {iteration_to_show}",
                            height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show clustering accuracy
            if not show_true_clusters:
                from sklearn.metrics import adjusted_rand_score
                predicted_labels = np.argmax(responsibilities, axis=1)
                ari_score = adjusted_rand_score(true_labels, predicted_labels)
                st.metric("Clustering Accuracy (ARI)", f"{ari_score:.3f}")

elif demo_type == "🧩 Missing Data Problem":
    st.markdown("## 🧩 EM for Missing Data Problems")
    
    st.markdown("""
    <div class="concept-box">
        <h4>🎯 The Missing Data Challenge</h4>
        <p><strong>Problem:</strong> You have a dataset where some values are missing randomly</p>
        <p><strong>Traditional Approach:</strong> Delete rows with missing data (wasteful!)</p>
        <p><strong>EM Approach:</strong> Treat missing values as hidden variables and estimate them</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🎛️ Missing Data Simulation")
        
        n_samples = st.slider("Number of Samples", 100, 500, 200)
        missing_rate = st.slider("Missing Data Rate", 0.1, 0.5, 0.3)
        true_correlation = st.slider("True Correlation", 0.0, 0.9, 0.7)
        
        # Generate complete bivariate normal data
        np.random.seed(42)
        mean = [0, 0]
        cov = [[1, true_correlation], [true_correlation, 1]]
        complete_data = np.random.multivariate_normal(mean, cov, n_samples)
        
        # Introduce missing data
        missing_mask = np.random.random((n_samples, 2)) < missing_rate
        incomplete_data = complete_data.copy()
        incomplete_data[missing_mask] = np.nan
        
        # Calculate statistics
        complete_cases = ~np.any(missing_mask, axis=1)
        n_complete = np.sum(complete_cases)
        n_missing = n_samples - n_complete
        
        st.markdown("#### 📊 Data Statistics")
        st.metric("Complete Cases", f"{n_complete}/{n_samples}")
        st.metric("Missing Cases", n_missing)
        st.metric("Data Loss", f"{(n_missing/n_samples)*100:.1f}%")
    
    with col2:
        # Compare different approaches
        methods_comparison = {}
        
        # 1. Complete case analysis (listwise deletion)
        complete_data_subset = complete_data[complete_cases]
        if len(complete_data_subset) > 1:
            complete_case_corr = np.corrcoef(complete_data_subset.T)[0, 1]
            complete_case_mean = np.mean(complete_data_subset, axis=0)
            complete_case_cov = np.cov(complete_data_subset.T)
        else:
            complete_case_corr = 0
            complete_case_mean = [0, 0]
            complete_case_cov = np.eye(2)
        
        methods_comparison['Complete Case Analysis'] = {
            'correlation': complete_case_corr,
            'mean': complete_case_mean,
            'n_used': len(complete_data_subset)
        }
        
        # 2. Simple EM for missing data
        def em_missing_data(data, max_iters=50, tol=1e-6):
            """Simple EM for bivariate normal with missing data"""
            data_em = data.copy()
            n, p = data.shape
            
            # Initialize with available data means
            mu = np.nanmean(data_em, axis=0)
            sigma = np.nancov(data_em.T)
            
            for iteration in range(max_iters):
                mu_old = mu.copy()
                
                # E-step: Fill in missing values with conditional expectations
                for i in range(n):
                    if np.isnan(data_em[i, 0]) and not np.isnan(data_em[i, 1]):
                        # X1 missing, X2 observed
                        data_em[i, 0] = mu[0] + sigma[0, 1]/sigma[1, 1] * (data_em[i, 1] - mu[1])
                    elif not np.isnan(data_em[i, 0]) and np.isnan(data_em[i, 1]):
                        # X2 missing, X1 observed
                        data_em[i, 1] = mu[1] + sigma[1, 0]/sigma[0, 0] * (data_em[i, 0] - mu[0])
                    elif np.isnan(data_em[i, 0]) and np.isnan(data_em[i, 1]):
                        # Both missing - use current means
                        data_em[i, 0] = mu[0]
                        data_em[i, 1] = mu[1]
                
                # M-step: Update parameters
                mu = np.mean(data_em, axis=0)
                sigma = np.cov(data_em.T)
                
                # Check convergence
                if np.linalg.norm(mu - mu_old) < tol:
                    break
            
            return data_em, mu, sigma
        
        # Apply EM
        em_data, em_mean, em_cov = em_missing_data(incomplete_data)
        em_correlation = em_cov[0, 1] / np.sqrt(em_cov[0, 0] * em_cov[1, 1])
        
        methods_comparison['EM Imputation'] = {
            'correlation': em_correlation,
            'mean': em_mean,
            'n_used': n_samples
        }
        
        # True values for comparison
        methods_comparison['True Values'] = {
            'correlation': true_correlation,
            'mean': [0, 0],
            'n_used': n_samples
        }
        
        # Visualization
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=[
                               'Original Complete Data',
                               'Data with Missing Values',
                               'Complete Case Analysis',
                               'EM Imputation Result'
                           ])
        
        # Original data
        fig.add_trace(go.Scatter(
            x=complete_data[:, 0], y=complete_data[:, 1],
            mode='markers', name='Complete Data',
            marker=dict(color='blue', size=4, opacity=0.6)
        ), row=1, col=1)
        
        # Data with missing values
        observed_mask = ~np.any(missing_mask, axis=1)
        fig.add_trace(go.Scatter(
            x=complete_data[observed_mask, 0], y=complete_data[observed_mask, 1],
            mode='markers', name='Observed',
            marker=dict(color='green', size=4)
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=complete_data[~observed_mask, 0], y=complete_data[~observed_mask, 1],
            mode='markers', name='Missing',
            marker=dict(color='red', size=4, symbol='x')
        ), row=1, col=2)
        
        # Complete case analysis
        if len(complete_data_subset) > 0:
            fig.add_trace(go.Scatter(
                x=complete_data_subset[:, 0], y=complete_data_subset[:, 1],
                mode='markers', name='Complete Cases Only',
                marker=dict(color='orange', size=4)
            ), row=2, col=1)
        
        # EM imputation result
        fig.add_trace(go.Scatter(
            x=em_data[observed_mask, 0], y=em_data[observed_mask, 1],
            mode='markers', name='Observed',
            marker=dict(color='blue', size=4)
        ), row=2, col=2)
        
        fig.add_trace(go.Scatter(
            x=em_data[~observed_mask, 0], y=em_data[~observed_mask, 1],
            mode='markers', name='EM Imputed',
            marker=dict(color='red', size=4, symbol='diamond')
        ), row=2, col=2)
        
        fig.update_layout(height=600, title="Missing Data Handling Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Results table
        st.markdown("#### 📊 Method Comparison")
        
        comparison_df = pd.DataFrame({
            'Method': list(methods_comparison.keys()),
            'Estimated Correlation': [f"{methods_comparison[method]['correlation']:.3f}" 
                                    for method in methods_comparison.keys()],
            'Sample Size Used': [methods_comparison[method]['n_used'] 
                               for method in methods_comparison.keys()],
            'Correlation Error': [f"{abs(methods_comparison[method]['correlation'] - true_correlation):.3f}" 
                                for method in methods_comparison.keys()]
        })
        
        st.dataframe(comparison_df, hide_index=True)
        
        # Key insights
        st.markdown("""
        #### 🔍 Key Insights:
        - **Complete Case Analysis**: Uses only complete observations, may be biased and wasteful
        - **EM Imputation**: Uses all available information, typically more accurate
        - **Sample Size**: EM uses full dataset, complete case analysis loses data
        - **Bias**: EM estimates are usually closer to true parameters
        """)

elif demo_type == "⚖️ Side-by-Side Comparison":
    st.markdown("## ⚖️ MLE vs EM: Direct Comparison")
    
    st.markdown("""
    <div class="concept-box">
        <h4>🎯 Head-to-Head Comparison</h4>
        <p>This demo shows the same statistical problem solved with both MLE (complete data) 
        and EM (with hidden variables), so you can see the key differences.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Scenario: Estimating Two Gaussian Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="algorithm-box">
            <h4>🎯 MLE Approach</h4>
            <p><strong>Assumption:</strong> We know which data point came from which distribution</p>
            <p><strong>Method:</strong> Separate the data by known labels, apply MLE to each group</p>
            <p><strong>Advantage:</strong> Simple, direct calculation</p>
            <p><strong>Limitation:</strong> Requires labeled data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # MLE side controls
        st.markdown("#### 🎛️ MLE Setup")
        n_samples_mle = st.slider("Samples per group (MLE)", 50, 200, 100, key="mle_samples")
        
        # Generate labeled data for MLE
        np.random.seed(42)
        group1_data = np.random.normal(-1, 0.8, n_samples_mle)
        group2_data = np.random.normal(2, 1.2, n_samples_mle)
        
        # MLE estimates (analytical)
        mle_mu1, mle_sigma1 = np.mean(group1_data), np.std(group1_data, ddof=1)
        mle_mu2, mle_sigma2 = np.mean(group2_data), np.std(group2_data, ddof=1)
        
        st.markdown("#### 📊 MLE Results")
        st.metric("Group 1 Mean", f"{mle_mu1:.3f}", f"True: -1.0")
        st.metric("Group 1 Std", f"{mle_sigma1:.3f}", f"True: 0.8")
        st.metric("Group 2 Mean", f"{mle_mu2:.3f}", f"True: 2.0")
        st.metric("Group 2 Std", f"{mle_sigma2:.3f}", f"True: 1.2")
        
        # Plot MLE results
        fig_mle = go.Figure()
        
        # Data histograms
        fig_mle.add_trace(go.Histogram(
            x=group1_data, name='Group 1 Data', 
            histnorm='probability density', opacity=0.6, nbinsx=20
        ))
        fig_mle.add_trace(go.Histogram(
            x=group2_data, name='Group 2 Data', 
            histnorm='probability density', opacity=0.6, nbinsx=20
        ))
        
        # MLE fitted curves
        x_range = np.linspace(-5, 6, 200)
        mle_pdf1 = gaussian_pdf(x_range, mle_mu1, mle_sigma1)
        mle_pdf2 = gaussian_pdf(x_range, mle_mu2, mle_sigma2)
        
        fig_mle.add_trace(go.Scatter(
            x=x_range, y=mle_pdf1, mode='lines',
            name='MLE Fit Group 1', line=dict(color='blue', width=3)
        ))
        fig_mle.add_trace(go.Scatter(
            x=x_range, y=mle_pdf2, mode='lines',
            name='MLE Fit Group 2', line=dict(color='red', width=3)
        ))
        
        fig_mle.update_layout(title="MLE: Known Group Labels", height=400)
        st.plotly_chart(fig_mle, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="algorithm-box">
            <h4>🔄 EM Approach</h4>
            <p><strong>Assumption:</strong> We DON'T know which data point came from which distribution</p>
            <p><strong>Method:</strong> Iteratively estimate group assignments and parameters</p>
            <p><strong>Advantage:</strong> Works with unlabeled data</p>
            <p><strong>Challenge:</strong> More complex, may find local optima</p>
        </div>
        """, unsafe_allow_html=True)
        
        # EM side controls
        st.markdown("#### 🎛️ EM Setup")
        mixing_proportion = st.slider("True Mixing Proportion", 0.3, 0.7, 0.5, key="em_mixing")
        em_iterations = st.slider("EM Iterations", 5, 30, 15, key="em_iters")
        
        # Generate mixed (unlabeled) data for EM
        n_total = n_samples_mle * 2
        n_from_group1 = int(n_total * mixing_proportion)
        n_from_group2 = n_total - n_from_group1
        
        mixed_data = np.concatenate([
            np.random.normal(-1, 0.8, n_from_group1),
            np.random.normal(2, 1.2, n_from_group2)
        ])
        np.random.shuffle(mixed_data)  # Mix the data
        
        # Apply EM
        em_model = GaussianMixtureEM(n_components=2)
        em_model.fit(mixed_data, max_iters=em_iterations)
        
        final_em_state = em_model.history[-1]
        
        # Sort components by mean (for consistent comparison)
        sort_idx = np.argsort(final_em_state['means'])
        em_mu1 = final_em_state['means'][sort_idx[0]]
        em_mu2 = final_em_state['means'][sort_idx[1]]
        em_sigma1 = np.sqrt(final_em_state['variances'][sort_idx[0]])
        em_sigma2 = np.sqrt(final_em_state['variances'][sort_idx[1]])
        em_weight1 = final_em_state['weights'][sort_idx[0]]
        em_weight2 = final_em_state['weights'][sort_idx[1]]
        
        st.markdown("#### 📊 EM Results")
        st.metric("Component 1 Mean", f"{em_mu1:.3f}", f"True: -1.0")
        st.metric("Component 1 Std", f"{em_sigma1:.3f}", f"True: 0.8")
        st.metric("Component 2 Mean", f"{em_mu2:.3f}", f"True: 2.0")
        st.metric("Component 2 Std", f"{em_sigma2:.3f}", f"True: 1.2")
        st.metric("Mixing Weight", f"{em_weight1:.3f}", f"True: {mixing_proportion:.1f}")
        
        # Plot EM results
        fig_em = go.Figure()
        
        # Mixed data histogram
        fig_em.add_trace(go.Histogram(
            x=mixed_data, name='Mixed Data', 
            histnorm='probability density', opacity=0.5, nbinsx=30
        ))
        
        # EM fitted components
        x_range = np.linspace(-5, 6, 200)
        em_pdf1 = em_weight1 * gaussian_pdf(x_range, em_mu1, em_sigma1)
        em_pdf2 = em_weight2 * gaussian_pdf(x_range, em_mu2, em_sigma2)
        em_total = em_pdf1 + em_pdf2
        
        fig_em.add_trace(go.Scatter(
            x=x_range, y=em_pdf1, mode='lines',
            name='EM Component 1', line=dict(color='blue', width=2, dash='dash')
        ))
        fig_em.add_trace(go.Scatter(
            x=x_range, y=em_pdf2, mode='lines',
            name='EM Component 2', line=dict(color='red', width=2, dash='dash')
        ))
        fig_em.add_trace(go.Scatter(
            x=x_range, y=em_total, mode='lines',
            name='EM Total Mixture', line=dict(color='black', width=3)
        ))
        
        fig_em.update_layout(title="EM: Unknown Group Labels", height=400)
        st.plotly_chart(fig_em, use_container_width=True)
    
    # Comparison summary
    st.markdown("### 📊 Method Comparison Summary")
    
    comparison_data = {
        'Aspect': [
            'Data Requirement',
            'Group 1 Mean Error',
            'Group 2 Mean Error',
            'Group 1 Std Error',
            'Group 2 Std Error',
            'Computational Complexity',
            'Convergence'
        ],
        'MLE': [
            'Labeled data required',
            f'{abs(mle_mu1 - (-1.0)):.3f}',
            f'{abs(mle_mu2 - 2.0):.3f}',
            f'{abs(mle_sigma1 - 0.8):.3f}',
            f'{abs(mle_sigma2 - 1.2):.3f}',
            'O(n) - Linear',
            'Direct solution'
        ],
        'EM': [
            'Unlabeled data OK',
            f'{abs(em_mu1 - (-1.0)):.3f}',
            f'{abs(em_mu2 - 2.0):.3f}',
            f'{abs(em_sigma1 - 0.8):.3f}',
            f'{abs(em_sigma2 - 1.2):.3f}',
            'O(nkT) - Iterative',
            f'{len(em_model.history)} iterations'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, hide_index=True)
    
    # Key takeaways
    st.markdown("""
    <div class="comparison-box">
        <h4>🎯 Key Takeaways:</h4>
        <ul>
            <li><strong>MLE is simpler</strong> when you have complete, labeled data</li>
            <li><strong>EM is more powerful</strong> when dealing with hidden variables or missing labels</li>
            <li><strong>Accuracy:</strong> Both can achieve similar accuracy under ideal conditions</li>
            <li><strong>Flexibility:</strong> EM can handle more complex scenarios (missing data, latent variables)</li>
            <li><strong>Computational cost:</strong> MLE is faster, EM requires iteration</li>
            <li><strong>Robustness:</strong> MLE gives exact solution, EM may find local optima</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer with learning resources
st.markdown("---")
st.markdown("""
### 🎓 **Summary: MLE vs EM**

**When to use MLE:**
- You have complete, observed data
- You know the model structure
- You want the simplest, most direct solution
- Computational efficiency is important

**When to use EM:**
- You have missing or hidden data
- You're dealing with mixture models
- You need to cluster unlabeled data
- You're working with latent variable models

**Key Mathematical Insight:**
- **MLE:** Directly maximizes L(θ|observed data)
- **EM:** Iteratively maximizes L(θ|observed data, hidden variables)

**Remember:** EM is essentially a way to do MLE when you have incomplete information! 🌟

### 🚀 **Next Steps:**
- Try different parameter values to see how robust each method is
- Experiment with different amounts of data and see how performance changes  
- Consider real-world applications where you might need each approach
- Think about when you'd prefer simplicity (MLE) vs flexibility (EM)
""")