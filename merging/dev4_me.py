# catatatan: dev4_me
# masih terdapat error di interpolasi residu kriging dan idw -> salah penempatan penggunaannya, harusnya digunakand di loocv
# kemudian di run gwr nya dilakukan juga kriging dan idw untuk residunya , kemudian di merge kan

# import modules
import pandas as pd
import numpy as np  
import geopandas as gpd
from pathlib import Path
import xarray as xr
import rasterio
from rasterio.features import rasterize
from pathlib import Path
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from shapely.geometry import Point
import warnings
import rioxarray
from rasterio.warp import reproject, Resampling
from pyproj import CRS, Transformer
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime

from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, r2_score
import pykrige
from pykrige.ok import OrdinaryKriging
from pykrige import variogram_models

try:
    from pykrige.ok import OrdinaryKriging
    from pykrige import variogram_models
    KRIGING_AVAILABLE = True
    print("PyKrige is available for kriging interpolation.")
except ImportError:
    KRIGING_AVAILABLE = False
    print("Warning: PyKrige not available. Kriging interpolation will fall back to IDW.")
    print("To install PyKrige: pip install pykrige")

# Suppress ignorable warnings from geopandas/shapely
warnings.filterwarnings('ignore', 'GeoSeries.isna', UserWarning)
# Menonaktifkan peringatan yang tidak relevan untuk menjaga output tetap bersih
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# configure
# base directory for the project
BASE_DIR = Path(r"D:\Kuliah\Semester_03\Thesis\olah")
OUTPUT_DIR = BASE_DIR / "output"

# path input files
OBS_CSV_PATH = OUTPUT_DIR / "quality_control" / "data_hujan_bulanan_lolos_seleksi_timeseries.csv"
CMORPH_DIR = OUTPUT_DIR / "99_sample_gwr_buffer" / "cmorph_with_buffer"
CHIRPS_DIR = OUTPUT_DIR / "99_sample_gwr_buffer" / "chrips_with_buffer_shp"
DEM_PATH = BASE_DIR / "data" / "dem" / "DEM_Sulawesi_30m_geom_buffer_rescaled_masked_buffer_clipped.tif"
SULAWESI_SHP_PATH = BASE_DIR / "data" / "shp" / "sulawesi_buffered.shp"

# Membuat direktori output jika belum ada
GWR_RESULTS_DIR = OUTPUT_DIR / "gwr_dev_loocv_results_dev4_me"
GWR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# range analysis date
START_YEAR = 1998
END_YEAR = 1998

print(f"Configurations have been set. The output will be saved in {OUTPUT_DIR}")

# Variables used for each schema
scenarios = {
    1: {'name': 'OBS_CMORPH', 'predictors': ['cmorph']},
    2: {'name': 'OBS_CMORPH_DEM', 'predictors': ['cmorph', 'dem']},
    3: {'name': 'OBS_CHIRPS', 'predictors': ['chirps']},
    4: {'name': 'OBS_CHIRPS_DEM', 'predictors': ['chirps', 'dem']},
    5: {'name': 'OBS_CMORPH_CHIRPS', 'predictors': ['cmorph', 'chirps']},
    6: {'name': 'OBS_CMORPH_CHIRPS_DEM', 'predictors': ['cmorph', 'chirps', 'dem']}
}

# Membuat direktori untuk setiap skenario
for sc_id, sc_info in scenarios.items():
    scenario_dir = GWR_RESULTS_DIR / sc_info['name']
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    # Subdirectory untuk LOOCV results
    loocv_dir = scenario_dir / "loocv_results"
    loocv_dir.mkdir(parents=True, exist_ok=True)
    
    # Subdirectory untuk performance metrics
    performance_dir = scenario_dir / "performance_metrics"
    performance_dir.mkdir(parents=True, exist_ok=True)
    
    # Subdirectory untuk raster results
    raster_dir = scenario_dir / "raster_results"
    raster_dir.mkdir(parents=True, exist_ok=True)
    
    # Subdirectory untuk residual rasters
    residual_dir = scenario_dir / "residual_rasters"
    residual_dir.mkdir(parents=True, exist_ok=True)
    
    # Subdirectory untuk merged results
    merged_dir = scenario_dir / "merged_results"
    merged_dir.mkdir(parents=True, exist_ok=True)

#  Functions needed for GWR analysis
def load_and_prepare_observation_data(obs_csv_path):
    """
    Load observation data once and prepare it for analysis
    
    Returns:
    --------
    df_obs : DataFrame
        Observation data with datetime index and year/month columns
    """
    print("Loading and preparing observation data...")
    start_time = time.time()
    
    try:
        df_obs = pd.read_csv(obs_csv_path)
        df_obs['tanggal'] = pd.to_datetime(df_obs['tanggal'])
        
        # Add year and month columns for easy filtering
        df_obs['year'] = df_obs['tanggal'].dt.year
        df_obs['month'] = df_obs['tanggal'].dt.month
        
        # Pre-compute coordinates for efficiency
        df_obs['coords_4326'] = list(zip(df_obs['CURRENT LONGITUDE'], 
                                         df_obs['CURRENT LATITUDE']))
        
        load_time = time.time() - start_time
        print(f"Loaded {len(df_obs)} observation records in {load_time:.2f} seconds")
        print(f"Date range: {df_obs['tanggal'].min()} to {df_obs['tanggal'].max()}")
        
        return df_obs
        
    except FileNotFoundError:
        raise FileNotFoundError(f"ERROR: Observation CSV not found at {obs_csv_path}")
    except Exception as e:
        raise Exception(f"ERROR loading observation data: {e}")

def get_monthly_observations(df_obs, year, month):
    """
    Filter observations for a specific year and month
    
    Parameters:
    -----------
    df_obs : DataFrame
        Pre-loaded observation data with year and month columns
    year : int
        Target year
    month : int
        Target month
        
    Returns:
    --------
    DataFrame
        Filtered observations for the specified year-month
    """
    return df_obs[(df_obs['year'] == year) & (df_obs['month'] == month)].copy()

def format_time(seconds):
    """Format seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.2f} minutes"
    else:
        return f"{seconds/3600:.2f} hours"

def extract_values_at_coords(coords_xy, raster_path):
    """
    Safely extracts values from a raster at specified coordinates.
    """
    try:
        with rasterio.open(raster_path) as raster:
            return np.array([val[0] for val in raster.sample(coords_xy)])
    except Exception as e:
        print(f"     ERROR: Could not extract values from {Path(raster_path).name}. Error: {e}")
        return np.full(len(coords_xy), np.nan)

def find_raster_file(directory, pattern):
    """Finds a single file in a directory matching a glob pattern."""
    files = list(directory.glob(pattern))
    if files:
        return files[0]
    return None

def interpolate_residuals_using_idw(residuals, coords, prediction_grid, grid_coords, power=2):
    """
    Interpolasi residu menggunakan Inverse Distance Weighting (IDW)
    
    Parameters:
    -----------
    residuals : array
        Nilai residu di titik observasi
    coords : array
        Koordinat titik observasi (x, y)
    prediction_grid : tuple or None
        Not used in this implementation (kept for compatibility)
    grid_coords : array
        Koordinat grid untuk prediksi
    power : float
        Power parameter untuk IDW
    """
    print(f"     Performing IDW interpolation with power={power}...")
    
    # Hitung jarak antara titik grid dan titik observasi
    distances = cdist(grid_coords, coords)
    
    # Hindari pembagian dengan nol
    distances[distances == 0] = 1e-10
    
    # Hitung bobot IDW
    weights = 1 / (distances ** power)
    
    # Normalisasi bobot
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # Interpolasi
    interpolated_residuals = np.dot(weights, residuals)
    
    return interpolated_residuals

def interpolate_residuals_using_ok(residuals, coords, prediction_grid, grid_coords, 
                                  variogram_model='spherical', nlags=6, weight=False):
    """
    Interpolasi residu menggunakan Ordinary Kriging
    
    Parameters:
    -----------
    residuals : array
        Nilai residu di titik observasi
    coords : array
        Koordinat titik observasi (x, y)
    prediction_grid : tuple or None
        Not used in this implementation (kept for compatibility)
    grid_coords : array
        Koordinat grid untuk prediksi
    """
    if not KRIGING_AVAILABLE:
        print("     PyKrige not available. Using IDW instead of Ordinary Kriging...")
        interpolated_residuals = interpolate_residuals_using_idw(
            residuals, coords, None, grid_coords
        )
        kriging_variance = np.full_like(interpolated_residuals, np.nan)
        return interpolated_residuals, kriging_variance
    
    print(f"     Performing Ordinary Kriging with {variogram_model} variogram...")
    
    try:
        # Siapkan koordinat
        x_obs = coords[:, 0]
        y_obs = coords[:, 1]
        x_grid = grid_coords[:, 0]
        y_grid = grid_coords[:, 1]
        
        # Buat model Ordinary Kriging
        ok_model = OrdinaryKriging(
            x_obs, y_obs, residuals,
            variogram_model=variogram_model,
            nlags=nlags,
            weight=weight,
            enable_plotting=False,
            verbose=False
        )
        
        # Lakukan interpolasi
        interpolated_residuals, kriging_variance = ok_model.execute(
            'points', x_grid, y_grid
        )
        
        return interpolated_residuals, kriging_variance
        
    except Exception as e:
        print(f"     ERROR in Ordinary Kriging: {e}")
        print("     Falling back to IDW interpolation...")
        interpolated_residuals = interpolate_residuals_using_idw(
            residuals, coords, None, grid_coords
        )
        kriging_variance = np.full_like(interpolated_residuals, np.nan)
        return interpolated_residuals, kriging_variance

def fill_nodata_at_points(raster_path, points_coords, window_size=3):
    """
    Fills NoData values at specific point locations in a raster by averaging surrounding pixels.

    Parameters:
    -----------
    raster_path : str or Path
        Path to the raster file to be modified.
    points_coords : array
        An array of (x, y) coordinates where NoData values should be checked and filled.
    window_size : int
        The size of the square window (e.g., 3 for a 3x3 window) to use for averaging.
        Must be an odd number.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number.")

    points_filled = 0
    with rasterio.open(str(raster_path), 'r+') as src:
        nodata_val = src.nodata
        raster_array = src.read(1)
        
        # Iterate over each point
        for point in points_coords:
            # Convert world coordinates to pixel indices
            try:
                row, col = src.index(point[0], point[1])
            except IndexError:
                # Point is outside the raster bounds
                continue

            # Check if the pixel at the point location is NoData
            is_nodata = (np.isnan(nodata_val) and np.isnan(raster_array[row, col])) or \
                        (raster_array[row, col] == nodata_val)
            
            if is_nodata:
                # Define the window
                half_win = window_size // 2
                window = raster_array[
                    max(0, row - half_win):min(src.height, row + half_win + 1),
                    max(0, col - half_win):min(src.width, col + half_win + 1)
                ]

                # Calculate the mean of valid pixels in the window
                if np.isnan(nodata_val):
                    valid_pixels = window[~np.isnan(window)]
                else:
                    valid_pixels = window[window != nodata_val]
                
                if valid_pixels.size > 0:
                    fill_value = np.mean(valid_pixels)
                    # Update the raster array in memory
                    raster_array[row, col] = fill_value
                    points_filled += 1

        if points_filled > 0:
            # Write the modified array back to the file
            src.write(raster_array, 1)
            print(f"     Filled {points_filled} NoData value(s) at observation points in {Path(raster_path).name}")

def create_residual_rasters_and_merge(df_month_obs_clean, scenario, year, month, 
                                     final_obs_coords, stacked_da, sulawesi_shp_proj, 
                                     out_meta, raster_out_path, predictor_paths):
    """
    Membuat raster residu menggunakan berbagai metode interpolasi dan menggabungkannya
    dengan hasil prediksi GWR
    """
    print("     Creating residual interpolation rasters...")
    
    # Ambil residu dari LOOCV
    residual_column = f'RESIDUAL_PREDICTED_RAINFALL_LOOCV_{scenario["name"]}'
    residuals = df_month_obs_clean[residual_column].values
    
    # Hapus NaN dari residuals
    valid_residual_mask = ~np.isnan(residuals)
    if np.sum(valid_residual_mask) < 3:
        print("     Not enough valid residuals for interpolation. Skipping...")
        return
    
    residuals_clean = residuals[valid_residual_mask]
    coords_clean = final_obs_coords[valid_residual_mask]
    
    # Bersihkan koordinat duplikasi
    print("     Cleaning duplicate coordinates if any...")
    residuals_clean, coords_clean = clean_duplicate_coordinates(residuals_clean, coords_clean)
    
    if len(residuals_clean) < 3:
        print("     Not enough valid points after cleaning duplicates. Skipping...")
        return
    
    print(f"     Using {len(residuals_clean)} points for interpolation")
    
    # Siapkan grid untuk prediksi
    xx, yy = np.meshgrid(stacked_da.x.values, stacked_da.y.values)
    pixel_coords = np.array(list(zip(xx.flatten(), yy.flatten())))
    
    # Mask untuk area Sulawesi
    valid_pixel_mask = ~np.isnan(stacked_da.isel(band=0).to_numpy().flatten())
    pixel_coords_clean = pixel_coords[valid_pixel_mask]
    
    print(f"     Interpolating to {len(pixel_coords_clean)} grid points")
    
    # Direktori untuk menyimpan raster residu
    residual_dir = GWR_RESULTS_DIR / scenario['name'] / "residual_rasters"
    residual_dir.mkdir(parents=True, exist_ok=True)
    
    # Direktori untuk menyimpan raster gabungan
    merged_dir = GWR_RESULTS_DIR / scenario['name'] / "merged_results"
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    # Baca raster GWR yang sudah ada
    with rasterio.open(raster_out_path) as src:
        gwr_raster = src.read(1)
    
    # Metode interpolasi - hanya IDW dan OK
    interpolation_methods = {
        'IDW': lambda: (
            interpolate_residuals_using_idw(
                residuals_clean, coords_clean, None, pixel_coords_clean
            ), None
        ),
        'OK': lambda: interpolate_residuals_using_ok(
            residuals_clean, coords_clean, None, pixel_coords_clean
        )
    }
    
    # Lakukan interpolasi untuk setiap metode
    interpolation_results = {}
    
    for method_name, method_func in interpolation_methods.items():
        print(f"     Processing {method_name} interpolation...")
        
        try:
            result = method_func()
            
            # Handle different return types
            if isinstance(result, tuple):
                interpolated_residuals, kriging_variance = result
            else:
                interpolated_residuals = result
                kriging_variance = None
            
            if interpolated_residuals is None:
                print(f"     {method_name} interpolation skipped (no valid data)")
                continue
            
            # Simpan hasil untuk logging
            interpolation_results[method_name] = {
                'mean': np.nanmean(interpolated_residuals),
                'std': np.nanstd(interpolated_residuals),
                'min': np.nanmin(interpolated_residuals),
                'max': np.nanmax(interpolated_residuals)
            }
            
            # Buat raster residu
            full_residual_raster = np.full(xx.shape, np.nan, dtype=np.float32).flatten()
            full_residual_raster[valid_pixel_mask] = interpolated_residuals
            full_residual_raster = full_residual_raster.reshape(xx.shape)
            
            # Simpan raster residu
            residual_raster_path = residual_dir / f"Residual_{method_name}_{scenario['name']}_{year}_{month:02d}.tif"
            with rasterio.open(residual_raster_path, 'w', **out_meta) as dst:
                dst.write(full_residual_raster.astype(np.float32), 1)

            # ===== PERUBAHAN DIMULAI DI SINI =====
            # Verifikasi dan isi NoData di lokasi stasiun pada raster residu
            print(f"     Verifying and filling NoData at station locations in the residual raster for {method_name}...")
            fill_nodata_at_points(residual_raster_path, coords_clean)
            
            # Baca kembali raster residu yang mungkin sudah diperbaiki untuk proses penggabungan
            with rasterio.open(residual_raster_path) as src:
                full_residual_raster = src.read(1)
            # ===== PERUBAHAN SELESAI =====
            
            # Buat raster gabungan (GWR + Residual)
            merged_raster = gwr_raster + full_residual_raster
            merged_raster = np.maximum(merged_raster, 0)  # Pastikan tidak ada nilai negatif
            
            # Simpan raster gabungan
            merged_raster_path = merged_dir / f"GWR_plus_{method_name}_{scenario['name']}_{year}_{month:02d}.tif"
            with rasterio.open(merged_raster_path, 'w', **out_meta) as dst:
                dst.write(merged_raster.astype(np.float32), 1)
            
            # Simpan variance jika ada (untuk Kriging)
            if kriging_variance is not None and method_name == 'OK':
                full_variance_raster = np.full(xx.shape, np.nan, dtype=np.float32).flatten()
                full_variance_raster[valid_pixel_mask] = kriging_variance
                full_variance_raster = full_variance_raster.reshape(xx.shape)
                
                variance_raster_path = residual_dir / f"Variance_{method_name}_{scenario['name']}_{year}_{month:02d}.tif"
                with rasterio.open(variance_raster_path, 'w', **out_meta) as dst:
                    dst.write(full_variance_raster.astype(np.float32), 1)
            
            print(f"     {method_name} interpolation completed successfully.")
            
        except Exception as e:
            print(f"     ERROR in {method_name} interpolation: {e}")
            continue
    
    # Tampilkan ringkasan hasil interpolasi
    if interpolation_results:
        print("     ========== INTERPOLATION RESULTS SUMMARY ==========")
        for method, stats in interpolation_results.items():
            print(f"     {method:12s}: Mean={stats['mean']:6.2f}, Std={stats['std']:6.2f}, Range=[{stats['min']:6.2f}, {stats['max']:6.2f}]")
        print("     ===================================================")
    
    print("     Residual interpolation and merging completed.")

def evaluate_residual_interpolation(df_month_obs_clean, scenario, residuals_clean, coords_clean, predictor_paths=None):
    """
    Evaluasi cross-validation untuk interpolasi residu
    
    Parameters:
    -----------
    df_month_obs_clean : DataFrame
        Data observasi yang sudah dibersihkan
    scenario : dict
        Informasi scenario
    residuals_clean : array
        Residual values yang sudah dibersihkan
    coords_clean : array
        Koordinat yang sudah dibersihkan
    predictor_paths : dict
        Path ke file prediktor (tidak digunakan karena hanya IDW dan OK)
    """
    print("     Performing cross-validation for residual interpolation...")
    
    n_points = len(residuals_clean)
    if n_points < 5:
        print("     Not enough points for cross-validation.")
        return {}
    
    # Hasil cross-validation - hanya IDW dan OK
    cv_results = {
        'IDW': {'predictions': [], 'actuals': []}, 
        'OK': {'predictions': [], 'actuals': []}
    }
    
    # Leave-one-out cross-validation
    for i in range(n_points):
        # Data training (semua kecuali titik ke-i)
        train_coords = np.delete(coords_clean, i, axis=0)
        train_residuals = np.delete(residuals_clean, i, axis=0)
        
        # Data testing (titik ke-i)
        test_coord = coords_clean[i].reshape(1, -1)
        test_residual = residuals_clean[i]
        
        # IDW
        try:
            pred_idw = interpolate_residuals_using_idw(
                train_residuals, train_coords, None, test_coord
            )
            cv_results['IDW']['predictions'].append(pred_idw[0])
            cv_results['IDW']['actuals'].append(test_residual)
        except Exception as e:
            print(f"     IDW prediction failed for point {i}: {e}")
        
        # OK
        try:
            pred_ok, _ = interpolate_residuals_using_ok(
                train_residuals, train_coords, None, test_coord
            )
            cv_results['OK']['predictions'].append(pred_ok[0])
            cv_results['OK']['actuals'].append(test_residual)
        except Exception as e:
            print(f"     OK prediction failed for point {i}: {e}")
    
    # Hitung metrics untuk semua metode yang tersedia
    performance_metrics = {}
    for method, results in cv_results.items():
        if len(results['predictions']) > 0:
            preds = np.array(results['predictions'])
            actuals = np.array(results['actuals'])
            
            # Hapus NaN values
            valid_mask = ~(np.isnan(preds) | np.isnan(actuals))
            if np.sum(valid_mask) < 3:
                continue
                
            preds_clean = preds[valid_mask]
            actuals_clean = actuals[valid_mask]
            
            rmse = np.sqrt(mean_squared_error(actuals_clean, preds_clean))
            mae = mean_absolute_error(actuals_clean, preds_clean)
            
            # Hitung R2 dengan penanganan error
            try:
                r2 = r2_score(actuals_clean, preds_clean)
            except:
                r2 = np.nan
            
            performance_metrics[method] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'n_points': len(preds_clean)
            }
    
    return performance_metrics

def save_interpolation_summary(scenario_name, year, month, performance_metrics):
    """
    Simpan ringkasan hasil interpolasi residu
    """
    summary_dir = GWR_RESULTS_DIR / scenario_name / "interpolation_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = summary_dir / f"interpolation_summary_{year}_{month:02d}.txt"
    
    with open(summary_file, 'w') as f:
        f.write(f"Residual Interpolation Summary\n")
        f.write(f"Scenario: {scenario_name}\n")
        f.write(f"Year-Month: {year}-{month:02d}\n")
        f.write(f"="*50 + "\n\n")
        
        if performance_metrics:
            f.write("Performance Metrics (Cross-Validation):\n")
            f.write("-" * 40 + "\n")
            for method, metrics in performance_metrics.items():
                f.write(f"\n{method}:\n")
                f.write(f"  RMSE: {metrics['RMSE']:.4f}\n")
                f.write(f"  MAE:  {metrics['MAE']:.4f}\n")
                f.write(f"  R²:   {metrics['R2']:.4f}\n")
                f.write(f"  N:    {metrics['n_points']}\n")
        else:
            f.write("No performance metrics available.\n")
        
        f.write(f"\nFiles generated:\n")
        f.write(f"- Residual rasters (IDW, OK)\n")
        f.write(f"- Variance rasters (OK)\n") 
        f.write(f"- Merged rasters (GWR + residuals)\n")
    
    print(f"     Interpolation summary saved to {summary_file}")

# Fungsi untuk memvalidasi input data
def validate_interpolation_inputs(residuals, coords):
    """
    Validasi input data untuk interpolasi
    """
    if len(residuals) != len(coords):
        raise ValueError("Length mismatch between residuals and coordinates")
    
    if len(residuals) < 3:
        raise ValueError("At least 3 points required for interpolation")
    
    # Cek variabilitas
    if np.std(residuals) < 1e-10:
        print("     Warning: Very low residual variance detected")
    
    # Cek dan handle duplikasi koordinat
    unique_coords, unique_indices = np.unique(coords, axis=0, return_index=True)
    if len(unique_coords) < len(coords):
        print(f"     Warning: Found {len(coords) - len(unique_coords)} duplicate coordinates. Removing duplicates...")
        return residuals[unique_indices], coords[unique_indices]
    
    return residuals, coords

def clean_duplicate_coordinates(residuals, coords):
    """
    Membersihkan koordinat duplikasi dengan mengambil rata-rata nilai residu
    """
    if len(residuals) != len(coords):
        raise ValueError("Length mismatch between residuals and coordinates")
    
    # Konversi koordinat ke string untuk grouping
    coord_strings = [f"{x:.6f}_{y:.6f}" for x, y in coords]
    
    # Buat dictionary untuk mengelompokkan berdasarkan koordinat
    coord_groups = {}
    for i, coord_str in enumerate(coord_strings):
        if coord_str not in coord_groups:
            coord_groups[coord_str] = []
        coord_groups[coord_str].append(i)
    
    # Jika tidak ada duplikasi
    if len(coord_groups) == len(coords):
        return residuals, coords
    
    print(f"     Found {len(coords) - len(coord_groups)} duplicate coordinates. Averaging residuals...")
    
    # Buat array baru dengan koordinat unik
    unique_residuals = []
    unique_coords = []
    
    for coord_str, indices in coord_groups.items():
        if len(indices) == 1:
            # Tidak ada duplikasi
            unique_residuals.append(residuals[indices[0]])
            unique_coords.append(coords[indices[0]])
        else:
            # Ada duplikasi, ambil rata-rata residu
            avg_residual = np.mean([residuals[i] for i in indices])
            unique_residuals.append(avg_residual)
            unique_coords.append(coords[indices[0]])  # Ambil koordinat yang pertama
    
    return np.array(unique_residuals), np.array(unique_coords)

def run_gwr_analysis():
    """
    Main function to run GWR analysis with optimized data loading
    """
    print("\n==================== Starting GWR Analysis ====================")
    overall_start_time = time.time()
    
    # ===== LOAD DATA ONCE =====
    # Load observation data only once at the beginning
    try:
        df_obs = load_and_prepare_observation_data(OBS_CSV_PATH)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    print("\nLoading Sulawesi shapefile for clipping...")
    shapefile_start = time.time()
    try:
        sulawesi_shp = gpd.read_file(SULAWESI_SHP_PATH)
        print(f"Shapefile loaded in {time.time() - shapefile_start:.2f} seconds")
    except Exception as e:
        print(f"ERROR: Could not load shapefile {SULAWESI_SHP_PATH}. Error: {e}")
        return

    # Set up coordinate transformation once
    source_crs = CRS("EPSG:4326")
    target_crs = CRS("ESRI:102028")
    transformer_4326_to_proj = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    sulawesi_shp_proj = sulawesi_shp.to_crs(target_crs)

    print("\nStarting GWR Analysis...")
    for year in range(START_YEAR, END_YEAR + 1):
        year_start_time = time.time()
        print(f"\n==================== PROCESSING YEAR: {year} ====================")

        for month in range(1, 13):
            month_start_time = time.time()
            print(f"\n--- Processing Month: {month:02d} ---")
            
            # Use the optimized function to get monthly data
            df_month_obs = get_monthly_observations(df_obs, year, month)
            
            if len(df_month_obs) < 15:
                print(f"Skipping month {month:02d}: Insufficient observation points ({len(df_month_obs)}).")
                continue

            raster_files = {
                'dem': DEM_PATH, 
                'cmorph': find_raster_file(CMORPH_DIR, f"*_{year}_{month:02d}_*.nc"), 
                'chirps': find_raster_file(CHIRPS_DIR, f"*{year}.monthly_{month:02d}_*.nc")
            }
            
            # Use pre-computed coordinates
            obs_coords_4326 = df_month_obs['coords_4326'].tolist()
            obs_coords_projected = np.array(list(transformer_4326_to_proj.itransform(obs_coords_4326)))

            # 1. Muat CHIRPS sebagai template untuk bulan ini
            print("   Loading CHIRPS raster to create a master grid template...")
            chirps_template_path = find_raster_file(CHIRPS_DIR, f"*{year}.monthly_{month:02d}_*.nc")

            if not chirps_template_path:
                print(f"   WARNING: CHIRPS template file not found for {year}-{month:02d}. Skipping this month.")
                continue

            try:
                # 2. Buat master grid dengan me-reproyeksi CHIRPS ke CRS target
                chirps_ref = rioxarray.open_rasterio(chirps_template_path, masked=True).squeeze()
                if not chirps_ref.rio.crs:
                    chirps_ref.rio.write_crs("EPSG:4326", inplace=True)
                
                grid_template = chirps_ref.rio.reproject(target_crs, resampling=Resampling.bilinear)
                grid_template = grid_template.rio.clip(sulawesi_shp_proj.geometry, sulawesi_shp_proj.crs, drop=False, all_touched=True)
                print("   ✅ Master grid template created successfully.")
            except Exception as e:
                print(f"   ERROR: Could not create master grid from CHIRPS file. Skipping month. Error: {e}")
                continue

            for scenario_id, scenario in scenarios.items():
                scenario_start_time = time.time()
                print(f"  -> Running Scenario {scenario_id}: {scenario['name']}")
                
                try:
                    predictor_paths = {p: raster_files[p] for p in scenario['predictors']}
                    if any(path is None for path in predictor_paths.values()):
                        print(f"     Skipping scenario: Missing one or more required raster files.")
                        continue
                except KeyError as e:
                    print(f"     Skipping scenario: Predictor '{e}' not found in file list.")
                    continue

                print("     Extracting predictor values at station locations...")
                predictor_columns = []
                active_predictor_names = []
                for pred_name, pred_path in predictor_paths.items():
                    values = extract_values_at_coords(obs_coords_4326, pred_path)
                    predictor_columns.append(values)
                    active_predictor_names.append(pred_name)

                X_obs = np.column_stack(predictor_columns)
                y_obs = df_month_obs['RAINFALL DAY MM'].values.reshape(-1, 1)

                nan_mask = ~np.isnan(X_obs).any(axis=1) & ~np.isnan(y_obs).flatten()
                if np.sum(nan_mask) < 15:
                    print(f"     Skipping: Not enough valid points ({np.sum(nan_mask)}) after NaN removal.")
                    continue
                
                X_obs, y_obs = X_obs[nan_mask], y_obs[nan_mask]
                final_obs_coords = obs_coords_projected[nan_mask]
                df_month_obs_clean = df_month_obs[nan_mask].copy()

                variance = np.var(X_obs, axis=0)
                non_zero_var_mask = variance > 1e-6
                if not np.all(non_zero_var_mask):
                    X_obs = X_obs[:, non_zero_var_mask]
                    active_predictor_names = [name for i, name in enumerate(active_predictor_names) if non_zero_var_mask[i]]

                if X_obs.shape[1] == 0:
                    print("     Skipping: No valid predictors with non-zero variance remain.")
                    continue

                # standarisasi skala agar menghindari dominasi variabel skala besar dan model lebih stabil
                scaler_X, scaler_y = StandardScaler().fit(X_obs), StandardScaler().fit(y_obs)
                X_scaled, y_scaled = scaler_X.transform(X_obs), scaler_y.transform(y_obs)

                print("     Performing Leave-One-Out cross-validation...")
                loocv_start = time.time()
                loo_predictions = []
                for i in range(len(final_obs_coords)):
                    cal_coords, cal_y, cal_X = np.delete(final_obs_coords, i, axis=0), np.delete(y_scaled, i, axis=0), np.delete(X_scaled, i, axis=0)
                    pred_coords, pred_X = final_obs_coords[i].reshape(1, 2), X_scaled[i].reshape(1, -1)
                    try:
                        # train model gwr
                        bw_selector = Sel_BW(cal_coords.tolist(), cal_y, cal_X, fixed=False)
                        bw = bw_selector.search(bw_max=len(cal_coords) - 1)
                        loo_model = GWR(cal_coords.tolist(), cal_y, cal_X, bw, fixed=False)
                        loo_results = loo_model.fit()
                        # predict using the model
                        pred_results = loo_model.predict(pred_coords, pred_X, loo_results.scale, loo_results.resid_response)
                        loo_predictions.append(pred_results.predictions[0, 0])
                    except Exception as e:
                        loo_predictions.append(np.nan)

                num_loo_fails = np.isnan(loo_predictions).sum()
                if num_loo_fails > 0: 
                    print(f"     LOO failed for {num_loo_fails} of {len(final_obs_coords)} points.")
                print(f"     LOOCV completed in {format_time(time.time() - loocv_start)}")

                predicted_rainfall_mm = scaler_y.inverse_transform(np.array(loo_predictions).reshape(-1, 1)).flatten()
                residuals_predicted = y_obs.flatten() - predicted_rainfall_mm
                df_month_obs_clean[f'PREDICTED_RAINFALL_LOOCV_{scenario["name"]}'] = predicted_rainfall_mm
                df_month_obs_clean[f'RESIDUAL_PREDICTED_RAINFALL_LOOCV_{scenario["name"]}'] = residuals_predicted

                # Save LOOCV results
                loocv_dir = GWR_RESULTS_DIR / scenario['name'] / "loocv_results"
                loocv_filename = loocv_dir / f"loocv_{year}_{month:02d}.csv"
                df_month_obs_clean.to_csv(loocv_filename, index=False)
                        
                print("     Calibrating final model and preparing for raster prediction...")
                calib_start = time.time()
                try:
                    n_cal_points = len(final_obs_coords)
                    final_bw_selector = Sel_BW(final_obs_coords.tolist(), y_scaled, X_scaled, fixed=False)
                    final_bw = final_bw_selector.search(bw_max=n_cal_points - 1)
                    final_model_obj = GWR(final_obs_coords.tolist(), y_scaled, X_scaled, final_bw, fixed=False)
                    final_results = final_model_obj.fit()
                    print(f"     Model calibration completed in {format_time(time.time() - calib_start)}")
                except Exception as e:
                    print(f"     Could not calibrate final GWR model. Skipping raster prediction. Error: {e}")
                    continue
                
                try:
                    stacked_predictors_for_grid = []
                    for pred_name in active_predictor_names:
                        print(f"   Processing {pred_name}...")
                        rds = rioxarray.open_rasterio(predictor_paths[pred_name], masked=True).squeeze()
                        
                        # Pastikan CRS yang benar berdasarkan jenis data
                        if not rds.rio.crs:
                            if pred_name == 'dem':
                                rds.rio.write_crs(target_crs, inplace=True)  # DEM sudah dalam target CRS
                            else:
                                rds.rio.write_crs("EPSG:4326", inplace=True)  # CMORPH/CHIRPS dalam WGS84

                        if 'band' not in rds.coords: 
                            rds = rds.expand_dims(band=[pred_name])
                        rds_reprojected = rds.rio.reproject_match(grid_template, resampling=Resampling.bilinear)
                        stacked_predictors_for_grid.append(rds_reprojected)

                    stacked_da = xr.concat(stacked_predictors_for_grid, dim="band")
                    # stacked_da = xr.concat(stacked_predictors_for_grid, dim="band").rio.reproject(target_crs, resampling=Resampling.bilinear)
                    # stacked_da = stacked_da_original.rio.reproject_match(grid_template, resampling=Resampling.bilinear)
                    stacked_da = stacked_da.rio.clip(sulawesi_shp_proj.geometry, sulawesi_shp_proj.crs, drop=False, all_touched=True)
                except Exception as e:
                    print(f"     Could not create prediction grid. Skipping raster prediction. Error: {e}")
                    continue
                
                pixel_predictors = stacked_da.data.reshape(stacked_da.shape[0], -1).T
                valid_pixel_mask = ~np.isnan(pixel_predictors).any(axis=1)
                pixel_predictors_clean = pixel_predictors[valid_pixel_mask]

                if len(pixel_predictors_clean) == 0:
                    print("     No valid pixels found to predict on. Skipping raster creation.")
                    continue

                xx, yy = np.meshgrid(stacked_da.x.values, stacked_da.y.values)
                pixel_coords = np.array(list(zip(xx.flatten(), yy.flatten())))
                pixel_coords_clean = pixel_coords[valid_pixel_mask]

                print(f"     Predicting values for {len(pixel_coords_clean)} pixels...")
                raster_pred_start = time.time()

                chunk_size = n_cal_points - 1
                predictions_scaled = np.array([])

                for i in range(0, len(pixel_coords_clean), chunk_size):
                    coords_chunk = pixel_coords_clean[i:i+chunk_size]
                    predictors_chunk = pixel_predictors_clean[i:i+chunk_size]

                    if len(predictors_chunk) == 0: 
                        continue

                    predictors_chunk_scaled = scaler_X.transform(predictors_chunk)

                    pred_results_chunk = final_model_obj.predict(coords_chunk, predictors_chunk_scaled, final_results.scale, final_results.resid_response)
                    predictions_scaled = np.append(predictions_scaled, pred_results_chunk.predictions.flatten())

                full_raster_scaled = np.full(xx.shape, np.nan, dtype=np.float32).flatten()
                full_raster_scaled[valid_pixel_mask] = predictions_scaled
                full_raster_scaled = full_raster_scaled.reshape(xx.shape)
                predicted_raster_mm = scaler_y.inverse_transform(full_raster_scaled.reshape(-1, 1)).reshape(xx.shape)

                out_meta = {
                    'driver': 'GTiff', 
                    'height': stacked_da.shape[1], 
                    'width': stacked_da.shape[2], 
                    'transform': stacked_da.rio.transform(), 
                    'crs': stacked_da.rio.crs, 
                    'count': 1, 
                    'dtype': 'float32', 
                    'nodata': np.nan
                }
                predicted_raster_mm[np.isnan(stacked_da.isel(band=0).to_numpy())] = out_meta['nodata']
                predicted_raster_mm = np.maximum(predicted_raster_mm, 0) # Pastikan tidak ada nilai negatif
                
                print(f"     Raster prediction completed in {format_time(time.time() - raster_pred_start)}")
                
                # Simpan raster hasil
                raster_dir = GWR_RESULTS_DIR / scenario['name'] / "raster_results"
                raster_out_path = raster_dir / f"GWR_Predicted_Rainfall_{scenario['name']}_{year}_{month:02d}.tif"
                with rasterio.open(raster_out_path, 'w', **out_meta) as dst: 
                    dst.write(predicted_raster_mm.astype(np.float32), 1)

                # fill nodata pixels with predicted values from observation points
                print("     Verifying pixel values at observation points and filling if necessary...")
                points_to_fill = []

                with rasterio.open(raster_out_path) as src:
                    for i, coord in enumerate(final_obs_coords):
                        value_at_point = list(src.sample([coord]))[0][0]
                        
                        if np.isnan(value_at_point) or value_at_point == src.nodata:
                            print(f"     Found NoData at station {i}. Predicting with final calibrated model.")
                            pred_coord_single = coord.reshape(1, 2)
                            pred_X_single_scaled = X_scaled[i].reshape(1, -1)
                            
                            pred_results_single = final_model_obj.predict(
                                pred_coord_single, 
                                pred_X_single_scaled, 
                                final_results.scale, 
                                final_results.resid_response
                            )
                            
                            predicted_value_scaled = pred_results_single.predictions[0, 0]
                            fill_value_unscaled = scaler_y.inverse_transform(np.array([[predicted_value_scaled]]))
                            fill_value = fill_value_unscaled[0, 0]
                            fill_value = np.maximum(fill_value, 0)  # Ensure non-negative
                            
                            point_geom = {'type': 'Point', 'coordinates': tuple(coord)}
                            
                            if not np.isnan(fill_value):
                                points_to_fill.append({'geometry': point_geom, 'properties': {'value': fill_value}})

                if points_to_fill:
                    print(f"     Found {len(points_to_fill)} observation point(s) with no data. Burning predicted values into raster...")
                    with rasterio.open(raster_out_path, 'r+') as dst:
                        raster_array = dst.read(1)
                        shapes = [(item['geometry'], item['properties']['value']) for item in points_to_fill]
                        
                        burned_array = rasterio.features.rasterize(
                            shapes=shapes,
                            out_shape=dst.shape,
                            transform=dst.transform,
                            fill=0,
                            merge_alg=rasterio.enums.MergeAlg.replace,
                            dtype=np.float32
                        )
                        
                        np.copyto(raster_array, burned_array, where=(burned_array != 0))
                        dst.write(raster_array, 1)
                else:
                    print("     All observation points have valid pixel values. No filling needed.")

                # Buat raster interpolasi residu dan gabungan
                residual_start = time.time()
                create_residual_rasters_and_merge(
                    df_month_obs_clean, scenario, year, month, 
                    final_obs_coords, stacked_da, sulawesi_shp_proj, 
                    out_meta, raster_out_path, predictor_paths
                )
                print(f"     Residual interpolation completed in {format_time(time.time() - residual_start)}")
                
                # Evaluasi interpolasi residu (opsional)
                residual_column = f'RESIDUAL_PREDICTED_RAINFALL_LOOCV_{scenario["name"]}'
                residuals = df_month_obs_clean[residual_column].values
                valid_residual_mask = ~np.isnan(residuals)
                
                if np.sum(valid_residual_mask) >= 5:
                    residuals_clean = residuals[valid_residual_mask]
                    coords_clean = final_obs_coords[valid_residual_mask]
                    
                    try:
                        # Validasi input
                        validate_interpolation_inputs(residuals_clean, coords_clean)
                        
                        # Evaluasi performance
                        performance_metrics = evaluate_residual_interpolation(
                            df_month_obs_clean, scenario, residuals_clean, coords_clean, predictor_paths
                        )
                        
                        # Simpan performance metrics
                        if performance_metrics:
                            performance_dir = GWR_RESULTS_DIR / scenario['name'] / "performance_metrics"
                            performance_file = performance_dir / f"residual_interpolation_performance_{year}_{month:02d}.csv"
                            
                            perf_df = pd.DataFrame(performance_metrics).T
                            perf_df.to_csv(performance_file)
                            print(f"     Residual interpolation performance saved to {performance_file}")
                        
                        # Simpan summary
                        save_interpolation_summary(scenario['name'], year, month, performance_metrics)
                        
                    except Exception as e:
                        print(f"     Error in residual interpolation evaluation: {e}")

                scenario_time = time.time() - scenario_start_time
                print(f"     Scenario complete in {format_time(scenario_time)}. Outputs saved to {GWR_RESULTS_DIR / scenario['name']}")

            month_time = time.time() - month_start_time
            print(f"\n   Month {month:02d} processing completed in {format_time(month_time)}")

        year_time = time.time() - year_start_time
        print(f"\n Year {year} processing completed in {format_time(year_time)}")

    overall_time = time.time() - overall_start_time
    print(f"\n\n==================== GWR Analysis Finished ====================")
    print(f"Total processing time: {format_time(overall_time)}")
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Execute the script ---
if __name__ == "__main__":
    run_gwr_analysis()