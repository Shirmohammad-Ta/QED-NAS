import numpy as np
import h5py
import json
from scipy.sparse import lil_matrix, csr_matrix, dok_matrix
from scipy.sparse.linalg import eigs
import quimb as qu
import matplotlib.pyplot as plt
import logging
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('default')
sns.set_palette("husl")
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']


class QuantumDatasetCreator:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.systems = []

    def build_hamiltonian_matrix(self, params):
        try:
            if params['model'] == 'TFIM':
                return self.build_tfim_hamiltonian(params)
            elif params['model'] == 'Heisenberg':
                return self.build_heisenberg_hamiltonian(params)
            elif params['model'] == 'ToricCode':
                return self.build_toric_code_hamiltonian(params)
        except Exception as e:
            logger.error(f"Error building Hamiltonian: {e}")
            return None

    def build_tfim_hamiltonian(self, params):
        n_sites = params['lattice_size']
        H = dok_matrix((2 ** n_sites, 2 ** n_sites), dtype=float)

        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        identity = np.eye(2)

        for i in range(n_sites):
            # Transverse field term
            op_list = [identity] * n_sites
            op_list[i] = sigma_x
            H_term = qu.kron(*op_list).real
            H -= params['h'] * H_term

            # Ising interaction term
            if i < n_sites - 1 or params.get('boundary', 'open') == 'periodic':
                j = (i + 1) % n_sites
                if j != i:
                    op_list = [identity] * n_sites
                    op_list[i] = sigma_z
                    op_list[j] = sigma_z
                    H_term = qu.kron(*op_list).real
                    H -= params['J'] * H_term

        return csr_matrix(H)

    def build_heisenberg_hamiltonian(self, params):
        n_sites = params['lattice_size']
        H = dok_matrix((2 ** n_sites, 2 ** n_sites), dtype=float)

        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        identity = np.eye(2)

        for i in range(n_sites):
            j = (i + 1) % n_sites
            if j != i or params.get('boundary', 'open') == 'periodic':
                # XX interaction
                op_list = [identity] * n_sites
                op_list[i] = sigma_x
                op_list[j] = sigma_x
                H_term = qu.kron(*op_list).real
                H += params['J'] * H_term

                # YY interaction
                op_list = [identity] * n_sites
                op_list[i] = sigma_y
                op_list[j] = sigma_y
                H_term = qu.kron(*op_list).real
                H += params['J'] * H_term

                # ZZ interaction
                op_list = [identity] * n_sites
                op_list[i] = sigma_z
                op_list[j] = sigma_z
                H_term = qu.kron(*op_list).real
                H += params['delta'] * params['J'] * H_term

        return csr_matrix(H)

    def build_toric_code_hamiltonian(self, params):
        L = params['lattice_size']
        n_sites = 2 * L * L
        H = dok_matrix((2 ** n_sites, 2 ** n_sites), dtype=float)

        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        identity = np.eye(2)

        for x in range(L):
            for y in range(L):
                # Star operators (vertex terms)
                vertices = [
                    (x, y, 0), (x, y, 1),
                    ((x - 1) % L, y, 0), (x, (y - 1) % L, 1)
                ]
                A_op = self.build_toric_operator(vertices, sigma_x, identity, n_sites)
                H -= params.get('J_star', 1.0) * A_op

                # Plaquette operators
                plaquettes = [
                    (x, y, 0), (x, y, 1),
                    ((x + 1) % L, y, 0), (x, (y + 1) % L, 1)
                ]
                B_op = self.build_toric_operator(plaquettes, sigma_z, identity, n_sites)
                H -= params.get('J_plaquette', 1.0) * B_op

        return csr_matrix(H)

    def build_toric_operator(self, sites, pauli_matrix, identity, total_sites):
        op_list = [identity] * total_sites
        for site in sites:
            if len(site) == 3:
                x, y, type_idx = site
                idx = 2 * (x * len(site) + y) + type_idx
                if idx < total_sites:
                    op_list[idx] = pauli_matrix
        return qu.kron(*op_list).real

    def calculate_reduced_density_matrix(self, ground_state, params):
        n_sites = params['lattice_size']
        if params['model'] == 'ToricCode':
            n_sites = 2 * n_sites * n_sites
        half_size = n_sites // 2

        psi = ground_state.reshape([2 ** half_size, 2 ** (n_sites - half_size)])
        rho_reduced = psi @ psi.conj().T

        return rho_reduced

    def generate_hamiltonian_parameters(self, num_samples=100):
        parameters_list = []

        for i in range(num_samples):
            model_type = np.random.choice(['TFIM', 'Heisenberg', 'ToricCode'],
                                          p=[0.4, 0.4, 0.2])

            if model_type == 'TFIM':
                params = {
                    'model': 'TFIM',
                    'J': 1.0,
                    'h': np.random.uniform(0.1, 3.0),
                    'lattice_size': 4,
                    'boundary': np.random.choice(['open', 'periodic'])
                }
            elif model_type == 'Heisenberg':
                params = {
                    'model': 'Heisenberg',
                    'J': 1.0,
                    'delta': np.random.uniform(0.1, 3.0),
                    'lattice_size': 4,
                    'boundary': np.random.choice(['open', 'periodic'])
                }
            else:  # ToricCode
                params = {
                    'model': 'ToricCode',
                    'J_star': 1.0,
                    'J_plaquette': 1.0,
                    'lattice_size': 2,
                    'boundary': 'periodic'
                }

            parameters_list.append(params)

        return parameters_list

    def extract_proxy_features(self, spectrum):
        try:
            non_zero_spectrum = spectrum[spectrum > 1e-10]
            if len(non_zero_spectrum) == 0:
                return None

            entropy = -np.sum(non_zero_spectrum * np.log(non_zero_spectrum))
            log_spectrum = np.log(non_zero_spectrum)
            mean = np.mean(log_spectrum)
            variance = np.var(log_spectrum)
            skewness = np.mean((log_spectrum - mean) ** 3) / (variance ** 1.5 + 1e-10)

            return {
                'entropy': float(entropy),
                'mean': float(mean),
                'variance': float(variance),
                'skewness': float(skewness)
            }
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def create_dataset(self, num_samples=30, output_file='quantum_dataset.h5'):
        # تولید پارامترهای هامیلتونی به صورت تصادفی
        hamiltonian_params = self.generate_hamiltonian_parameters(num_samples)

        with h5py.File(output_file, 'w') as f:
            params_group = f.create_group('hamiltonian_parameters')
            features_group = f.create_group('proxy_features')

            successful_samples = 0
            model_counts = {'TFIM': 0, 'Heisenberg': 0, 'ToricCode': 0}

            for i, params in enumerate(hamiltonian_params):
                try:
                    logger.info(f"Processing sample {i + 1}/{len(hamiltonian_params)} - Model: {params['model']}")

                    # ذخیره پارامترها
                    params_group.create_dataset(
                        f'sample_{i}',
                        data=json.dumps(params)
                    )

                    # ساخت هامیلتونی و محاسبه حالت پایه
                    H = self.build_hamiltonian_matrix(params)
                    if H is None:
                        logger.warning(f"Failed to build Hamiltonian for sample {i}")
                        continue

                    # محاسبه حالت پایه
                    eigenvalues, eigenvectors = eigs(H, k=1, which='SR')
                    ground_state = eigenvectors[:, 0]

                    # محاسبه ماتریس چگالی کاهش یافته
                    rho_reduced = self.calculate_reduced_density_matrix(ground_state, params)

                    # استخراج طیف و ویژگی‌ها
                    spectrum = np.linalg.eigvalsh(rho_reduced)
                    features = self.extract_proxy_features(spectrum)

                    if features is None:
                        logger.warning(f"Failed to extract features for sample {i}")
                        continue

                    # ذخیره ویژگی‌ها
                    features_dataset = features_group.create_dataset(
                        f'sample_{i}',
                        data=np.array([
                            features['entropy'],
                            features['mean'],
                            features['variance'],
                            features['skewness']
                        ])
                    )

                    features_dataset.attrs['hamiltonian_params'] = json.dumps(params)
                    features_dataset.attrs['model_type'] = params['model']

                    successful_samples += 1
                    model_counts[params['model']] += 1

                    logger.info(f"Sample {i} completed - Entropy: {features['entropy']:.4f}")

                except Exception as e:
                    logger.error(f"Failed sample {i}: {e}")
                    continue

            logger.info(f"Successfully processed {successful_samples}/{num_samples} samples")
            logger.info(f"Model distribution: {model_counts}")

            return successful_samples, model_counts


def check_dataset(file_path='quantum_dataset.h5'):
    print("=" * 70)
    print("ANALYSIS OF QUANTUM DATASET")
    print("=" * 70)

    with h5py.File(file_path, 'r') as f:
        print("\n1. FILE STRUCTURE:")
        print("Groups in file:", list(f.keys()))

        model_counts = {'TFIM': 0, 'Heisenberg': 0, 'ToricCode': 0}

        if 'hamiltonian_parameters' in f:
            print(f"\n2. HAMILTONIAN PARAMETERS (First 5 samples):")
            print("-" * 50)
            for i in range(min(5, len(f['hamiltonian_parameters']))):
                params = json.loads(f['hamiltonian_parameters'][f'sample_{i}'][()])
                model_type = params['model']
                model_counts[model_type] += 1
                print(f"Sample {i}: {params}")

        if 'proxy_features' in f:
            print(f"\n3. PROXY FEATURES (First 5 samples):")
            print("-" * 50)
            for i in range(min(5, len(f['proxy_features']))):
                features = f['proxy_features'][f'sample_{i}'][()]
                params = json.loads(f['proxy_features'][f'sample_{i}'].attrs['hamiltonian_params'])
                print(f"Sample {i} ({params['model']}):")
                print(f"  Entropy    = {features[0]:.4f}")
                print(f"  Mean       = {features[1]:.4f}")
                print(f"  Variance   = {features[2]:.4f}")
                print(f"  Skewness   = {features[3]:.4f}")

        print(f"\n4. DATASET SUMMARY:")
        print("-" * 30)
        print(f"Total samples: {len(f['proxy_features'])}")
        print(f"Model distribution: {model_counts}")
        print(f"TFIM: {model_counts['TFIM']} samples")
        print(f"Heisenberg: {model_counts['Heisenberg']} samples")
        print(f"ToricCode: {model_counts['ToricCode']} samples")


def plot_entanglement_entropy(data):
    """پلات جداگانه برای آنتروپی درهم تنیدگی"""
    plt.figure(figsize=(10, 6))
    for model, color in zip(['TFIM', 'Heisenberg', 'ToricCode'], colors):
        if data[model]['entropy']:  # فقط اگر داده وجود دارد
            x = range(len(data[model]['entropy']))
            plt.plot(x, data[model]['entropy'], 'o-', label=model, color=color,
                     markersize=6, linewidth=2)
    plt.xlabel('Sample Index')
    plt.ylabel('Entanglement Entropy')
    plt.title('Entanglement Entropy Across Different Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_mean_spectrum(data):
    """پلات جداگانه برای میانگین طیف"""
    plt.figure(figsize=(10, 6))
    for model, color in zip(['TFIM', 'Heisenberg', 'ToricCode'], colors):
        if data[model]['mean']:  # فقط اگر داده وجود دارد
            x = range(len(data[model]['mean']))
            plt.plot(x, data[model]['mean'], 's-', label=model, color=color,
                     markersize=6, linewidth=2)
    plt.xlabel('Sample Index')
    plt.ylabel('Mean of log(spectrum)')
    plt.title('Mean of Log Spectrum Across Different Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_variance_spectrum(data):
    """پلات جداگانه برای واریانس"""
    plt.figure(figsize=(10, 6))
    for model, color in zip(['TFIM', 'Heisenberg', 'ToricCode'], colors):
        if data[model]['variance']:  # فقط اگر داده وجود دارد
            x = range(len(data[model]['variance']))
            plt.plot(x, data[model]['variance'], 'd-', label=model, color=color,
                     markersize=6, linewidth=2)
    plt.xlabel('Sample Index')
    plt.ylabel('Variance of log(spectrum)')
    plt.title('Variance of Log Spectrum Across Different Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_skewness_spectrum(data):
    """پلات جداگانه برای چولگی"""
    plt.figure(figsize=(10, 6))
    for model, color in zip(['TFIM', 'Heisenberg', 'ToricCode'], colors):
        if data[model]['skewness']:  # فقط اگر داده وجود دارد
            x = range(len(data[model]['skewness']))
            plt.plot(x, data[model]['skewness'], '^-', label=model, color=color,
                     markersize=6, linewidth=2)
    plt.xlabel('Sample Index')
    plt.ylabel('Skewness')
    plt.title('Skewness of Spectrum Across Different Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_mean_features_bar(data):
    """پلات میله‌ای برای میانگین ویژگی‌ها"""
    plt.figure(figsize=(12, 7))
    features_means = {}
    for model in ['TFIM', 'Heisenberg', 'ToricCode']:
        if data[model]['entropy']:  # فقط اگر داده وجود دارد
            features_means[model] = [
                np.mean(data[model]['entropy']),
                np.mean(data[model]['mean']),
                np.mean(data[model]['variance']),
                np.mean(data[model]['skewness'])
            ]

    if features_means:
        x = np.arange(4)
        width = 0.25
        for i, model in enumerate(features_means.keys()):
            plt.bar(x + i * width, features_means[model], width,
                    label=model, alpha=0.8, color=colors[i])

        plt.xlabel('Features')
        plt.ylabel('Mean Value')
        plt.title('Mean Feature Values by Model')
        plt.xticks(x + width, ['Entropy', 'Mean', 'Variance', 'Skewness'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_correlation_matrix(data):
    """پلات ماتریس همبستگی"""
    plt.figure(figsize=(10, 8))
    all_features = []
    for model in ['TFIM', 'Heisenberg', 'ToricCode']:
        for i in range(len(data[model]['entropy'])):
            all_features.append([
                data[model]['entropy'][i],
                data[model]['mean'][i],
                data[model]['variance'][i],
                data[model]['skewness'][i]
            ])

    if all_features:
        correlation_matrix = np.corrcoef(np.array(all_features).T)
        sns.heatmap(correlation_matrix,
                    annot=True,
                    cmap='coolwarm',
                    center=0,
                    xticklabels=['Entropy', 'Mean', 'Variance', 'Skewness'],
                    yticklabels=['Entropy', 'Mean', 'Variance', 'Skewness'])
        plt.title('Correlation Matrix of Entanglement Features')
        plt.tight_layout()
        plt.show()


def plot_dataset_analysis(file_path='quantum_dataset.h5'):
    try:
        with h5py.File(file_path, 'r') as f:
            if 'proxy_features' not in f:
                print("No features found in dataset")
                return

            data = {'TFIM': {'entropy': [], 'mean': [], 'variance': [], 'skewness': []},
                    'Heisenberg': {'entropy': [], 'mean': [], 'variance': [], 'skewness': []},
                    'ToricCode': {'entropy': [], 'mean': [], 'variance': [], 'skewness': []}}

            for i in range(len(f['proxy_features'])):
                features = f['proxy_features'][f'sample_{i}'][()]
                model_type = f['proxy_features'][f'sample_{i}'].attrs['model_type']

                if model_type in data:
                    data[model_type]['entropy'].append(features[0])
                    data[model_type]['mean'].append(features[1])
                    data[model_type]['variance'].append(features[2])
                    data[model_type]['skewness'].append(features[3])

            print("\n" + "=" * 70)
            print("VISUALIZATION OF QUANTUM DATASET")
            print("=" * 70)

            # نمایش جداگانه هر پلات
            print("\n1. Entanglement Entropy Plot...")
            plot_entanglement_entropy(data)

            print("\n2. Mean of Log Spectrum Plot...")
            plot_mean_spectrum(data)

            print("\n3. Variance of Log Spectrum Plot...")
            plot_variance_spectrum(data)

            print("\n4. Skewness Plot...")
            plot_skewness_spectrum(data)

            print("\n5. Mean Features Bar Chart...")
            plot_mean_features_bar(data)

            print("\n6. Correlation Matrix...")
            plot_correlation_matrix(data)

    except Exception as e:
        print(f"Error in plotting: {e}")


def main():
    print("QUANTUM DATASET CREATION STARTED")
    print("=" * 50)

    # استفاده از timestamp به عنوان seed برای تصادفی بودن در هر اجرا
    import time
    seed = int(time.time())
    print(f"Random seed: {seed}")

    creator = QuantumDatasetCreator(seed=seed)

    successful_samples, model_counts = creator.create_dataset(
        num_samples=30,
        output_file='quantum_dataset.h5'
    )

    print("\n" + "=" * 50)
    print("DATASET CREATION COMPLETED")
    print("=" * 50)
    print(f"Successfully created {successful_samples} samples")
    print(f"Model distribution: {model_counts}")

    # نمایش اطلاعات دیتاست
    check_dataset()

    # نمایش نمودارها به صورت جداگانه
    plot_dataset_analysis()


if __name__ == "__main__":
    main()