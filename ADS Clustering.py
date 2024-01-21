
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.optimize as opt
from cluster_tools import scaler
from errors import error_prop
import sklearn.metrics as skmet


def read_data(file):
    """
    Read data from a CSV file, clean it, and transpose it.

    Parameters:
    - file (str): Path to the CSV file.

    Returns:
    - df_clean (pd.DataFrame): Cleaned and transposed DataFrame.
    - df_t (pd.DataFrame): Transposed DataFrame.
    """
    df = pd.read_csv(file, index_col=0).iloc[:-7]
    df_clean = df.dropna(axis=1, how="all").dropna()
    df_clean.drop(columns=['Country Code', 'Series Code'], inplace=True)
    df_clean.columns = [col.split(' ')[0] for col in df_clean.columns]
    df_t = df_clean.transpose()
    return df_clean, df_t

def kmeans_cluster_alternative(nclusters, data):
    """
    Perform k-means clustering on the input data.

    Parameters:
    - nclusters (int): Number of clusters.
    - data (pd.DataFrame): Input data for clustering.

    Returns:
    - labels (np.ndarray): Cluster labels.
    - centroids (np.ndarray): Cluster centroids.
    """
    kmeans = KMeans(n_clusters=nclusters)
    labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_
    return labels, centroids

def poly(x, a, b, c):
    """
    Polynomial function used for curve fitting.

    Parameters:
    - x (np.ndarray): Input values.
    - a, b, c (float): Coefficients of the polynomial.

    Returns:
    - f (np.ndarray): Output values.
    """
    x = x - 2008
    f = a + b*x + c*x**2
    return f

def err_ranges(x, func, parameters, covariance, num_points=100):
    """
    Calculate error ranges for the given function.

    Parameters:
    - x (np.ndarray): Input values.
    - func (callable): Function for which error ranges are calculated.
    - parameters (np.ndarray): Function parameters.
    - covariance (np.ndarray): Covariance matrix of the parameters.
    - num_points (int): Number of points for error range.

    Returns:
    - low, up (np.ndarray): Lower and upper bounds of the error range.
    - x_values, y_values (np.ndarray): Input and output values for plotting.
    - low_values, up_values (np.ndarray): Lower and upper bounds for plotting.
    """
    sigma = error_prop(x, func, parameters, covariance)
    forecast = func(x, *parameters)

    low = forecast - sigma
    up = forecast + sigma

    x_values = np.linspace(min(x), max(x), num_points)
    y_values = func(x_values, *parameters)
    low_values = func(x_values, *parameters) - error_prop(x_values, func, parameters, covariance)
    up_values = func(x_values, *parameters) + error_prop(x_values, func, parameters, covariance)
    return low, up, x_values, y_values, low_values, up_values



def Plot_cluster_scatter(label, centroids):
    """
    Plot scatter plot with clustered data points and centroids.

    Parameters:
    - label (np.ndarray): Cluster labels.
    - centroids (np.ndarray): Cluster centroids.
    """
    plt.figure()
    cm = plt.cm.get_cmap('gist_rainbow')
    plt.scatter(growth_cluster['Total Debt Service'],
                growth_cluster["GNI Per Capita Growth"],
                s=30, marker="D", c=label, cmap=cm)
    plt.scatter(centroids[:,0],centroids[:,1], s=20,
                c="k", marker="d", label='Centroid')
    plt.title("GNI Per Capita vs Total Debt Service of China", fontsize=14)
    plt.xlabel("Total Debt Service", fontsize=12)
    plt.ylabel("GNI Per Capita", fontsize=12)
    for i in range(5):
        plt.scatter([], [], label=f'Cluster {i+1}',marker="D", c=cm(i/4), s=30)
    plt.legend()
    plt.tight_layout()
    plt.show()


def Plot_GNI_growth(China_growth):
    """
    Plot GNI Per Capita Growth over the years.

    Parameters:
    - China_growth (pd.DataFrame): DataFrame with GNI Per Capita Growth data.
    """
    plt.figure()
    plt.plot(China_growth.index, China_growth['GNI Per Capita Growth'], 
             color='lime', label='GNI Per Capita Growth', marker='o', 
             markerfacecolor='yellow')
    plt.xlabel("Years", fontsize=12)
    plt.ylabel("GNI Per Capita Growth (%)", fontsize=12)
    plt.title("GNI Per Capita Growth worldwide (2000-2022)", fontsize=14)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def Plot_Total_debt(China_growth):
    """
    Plot Total Debt Service over the years.

    Parameters:
    - China_growth (pd.DataFrame): DataFrame with Total Debt Service data.
    """
    plt.plot(China_growth.index, China_growth['Total Debt Service'], 
             color='purple', label='Total Debt Service',  
             marker='o', markerfacecolor='aqua')
    plt.xlabel("Years", fontsize=12)
    plt.ylabel('Total Debt Service (%)', fontsize=12)
    plt.title("Total Debt Service worldwide (2000-2022)", fontsize=14)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def Plot_GNI_growth_forecast(China_growth):
    """
    Plot GNI Per Capita Growth with forecast for China.

    Parameters:
    - China_growth (pd.DataFrame): DataFrame with GNI Per Capita Growth data.
    """
    China_growth = China_growth.reset_index()
    China_growth["Year"] = pd.to_numeric(China_growth["Year"])
    param_gni_growth, covar_gni_growth = opt.curve_fit(poly, China_growth["Year"], China_growth["GNI Per Capita Growth"])
    sigma_gni_growth = error_prop(China_growth["Year"], poly, param_gni_growth, covar_gni_growth)
    year_gni_growth = np.arange(2000, 2026)
    low_gni_growth, up_gni_growth, x_values_gni_growth, y_values_gni_growth, low_values_gni_growth, up_values_gni_growth = err_ranges(year_gni_growth, poly, param_gni_growth, covar_gni_growth)
    forecast_2025_gni_growth = poly(2025, *param_gni_growth)
    
    plt.figure()
    plt.plot(China_growth["Year"], China_growth["GNI Per Capita Growth"],
             label="GNI Per Capita Growth", color='olive')
    plt.plot(x_values_gni_growth, y_values_gni_growth, label="Forecast",
             color='red')
    plt.fill_between(x_values_gni_growth, low_values_gni_growth, up_values_gni_growth, color="pink", alpha=0.8, label="Confidence Range")
    plt.scatter([2025], [forecast_2025_gni_growth], c='blue', marker='D',
                label=f'2025 Prediction: {forecast_2025_gni_growth:.2f}%')
    plt.xlabel("Years", fontsize=12)
    plt.ylabel("GNI Per Capita Growth in %", fontsize=12)
    plt.title("GNI Per Capita Growth Forecast for China", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()


def Plot_Total_debt_forecast(China_growth):
    """
    Plot Total Debt Service with forecast for China.

    Parameters:
    - China_growth (pd.DataFrame): DataFrame with Total Debt Service data.
    """
    China_growth = China_growth.reset_index()
    China_growth["Year"] = pd.to_numeric(China_growth["Year"])
    China_growth["Total Debt Service"] = pd.to_numeric(China_growth["Total Debt Service"])
    param_total_debt, covar_total_debt = opt.curve_fit(poly, China_growth["Year"], China_growth["Total Debt Service"])
    sigma_total_debt = error_prop(China_growth["Year"], poly, param_total_debt, covar_total_debt)
    year_total_debt = np.arange(2000, 2026)
    low_total_debt, up_total_debt, x_values_total_debt, y_values_total_debt, low_values_total_debt, up_values_total_debt = err_ranges(year_total_debt, poly, param_total_debt, covar_total_debt)
    forecast_2025_total_debt = poly(2025, *param_total_debt)

    plt.figure()
    plt.plot(China_growth["Year"], China_growth["Total Debt Service"],
             label="Total Debt Service", color='green')
    plt.plot(x_values_total_debt, y_values_total_debt, label="Forecast",
             color='red')
    plt.fill_between(x_values_total_debt, low_values_total_debt, up_values_total_debt, color="pink", alpha=0.8, label="Confidence Range")
    plt.scatter([2025], [forecast_2025_total_debt], c='blue', marker='D',
                label=f'2025 Prediction: {forecast_2025_total_debt:.2f}%')
    plt.xlabel("Years", fontsize=12)
    plt.ylabel("Total Debt Service in %", fontsize=12)
    plt.title("Total Debt Service Forecast for China", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()


_, GNI_growth = read_data("GNI Per Capita Growth.csv")
_, Total_Debt = read_data("Total Debt Service.csv")

GNI_China = GNI_growth.loc[:, "China"].copy()
Debt_China = Total_Debt.loc["2000":"2022", "China"].copy()

GNI_China.index = GNI_China.index.map(str).str.replace('Series', '').str.strip()
Debt_China.index = Debt_China.index.map(str).str.replace('Series', '').str.strip()

GNI_China.index = pd.to_numeric(GNI_China.index)
Debt_China.index = pd.to_numeric(Debt_China.index)

China_growth = pd.merge(GNI_China, Debt_China, left_index=True, 
                        right_index=True, how="outer")
China_growth = China_growth.rename(columns={'China_x': "GNI Per Capita Growth",
                                            'China_y': "Total Debt Service"})
China_growth.index.name = 'Year'
China_growth = China_growth.apply(pd.to_numeric, errors='coerce')
China_growth = China_growth.dropna()

# The Normalized data
growth_cluster, df_min, df_max = scaler(China_growth[["GNI Per Capita Growth",
                                                      "Total Debt Service"]])

print("number of clusters   score")
for ncluster in range(2, 10):
    lab, cent = kmeans_cluster_alternative(ncluster, growth_cluster)
    print(ncluster, skmet.silhouette_score(growth_cluster, lab))

label, center = kmeans_cluster_alternative(5, growth_cluster)

Plot_cluster_scatter(label, center)

Plot_GNI_growth(China_growth)

Plot_Total_debt(China_growth)

Plot_GNI_growth_forecast(China_growth)

Plot_Total_debt_forecast(China_growth)






