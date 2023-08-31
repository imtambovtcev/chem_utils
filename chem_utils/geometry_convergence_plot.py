import sys
import matplotlib.pyplot as plt

def plot_convergence(file_path, save_path=None):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Lists to store data
    energy_changes = []
    rms_gradients = []
    max_gradients = []
    rms_steps = []
    max_steps = []
    
    # Lists to store tolerance
    energy_tolerance = []
    rms_gradient_tolerance = []
    max_gradient_tolerance = []
    rms_step_tolerance = []
    max_step_tolerance = []

    # Flags
    is_convergence_block = False

    # Loop through lines
    for line in lines:
        if "Geometry convergence" in line:
            is_convergence_block = True
            continue
        if is_convergence_block:
            values = line.split()
            if "Energy change" in line:
                energy_changes.append(float(values[2]))
                if values[4].replace('.', '', 1).isdigit():
                    energy_tolerance.append(float(values[4]))
            elif "RMS gradient" in line:
                rms_gradients.append(float(values[2]))
                if values[4].replace('.', '', 1).isdigit():
                    rms_gradient_tolerance.append(float(values[4]))
            elif "MAX gradient" in line:
                max_gradients.append(float(values[2]))
                if values[4].replace('.', '', 1).isdigit():
                    max_gradient_tolerance.append(float(values[4]))
            elif "RMS step" in line:
                rms_steps.append(float(values[2]))
                if values[4].replace('.', '', 1).isdigit():
                    rms_step_tolerance.append(float(values[4]))
            elif "MAX step" in line:
                max_steps.append(float(values[2]))
                if values[4].replace('.', '', 1).isdigit():
                    max_step_tolerance.append(float(values[4]))
            elif "........................................................" in line:
                is_convergence_block = False

    # Plot
    fig, axs = plt.subplots(5, 1, figsize=(10, 15))

    x_energy = range(len(energy_changes))
    axs[0].fill_between(x_energy, -abs(energy_tolerance[0]), abs(energy_tolerance[0]), color='orange', alpha=0.2)
    axs[0].plot(energy_changes, marker='o')
    axs[0].set_title('Energy Change Convergence')
    axs[0].set_ylabel('Energy Change')
    axs[0].grid(True)

    x_rms_gradient = range(len(rms_gradients))
    axs[1].fill_between(x_rms_gradient, -abs(rms_gradient_tolerance[0]), abs(rms_gradient_tolerance[0]), color='orange', alpha=0.2)
    axs[1].plot(rms_gradients, marker='o')
    axs[1].set_title('RMS Gradient Convergence')
    axs[1].set_ylabel('RMS Gradient')
    axs[1].grid(True)

    x_max_gradient = range(len(max_gradients))
    axs[2].fill_between(x_max_gradient, -abs(max_gradient_tolerance[0]), abs(max_gradient_tolerance[0]), color='orange', alpha=0.2)
    axs[2].plot(max_gradients, marker='o')
    axs[2].set_title('MAX Gradient Convergence')
    axs[2].set_ylabel('MAX Gradient')
    axs[2].grid(True)

    x_rms_step = range(len(rms_steps))
    axs[3].fill_between(x_rms_step, -abs(rms_step_tolerance[0]), abs(rms_step_tolerance[0]), color='orange', alpha=0.2)
    axs[3].plot(rms_steps, marker='o')
    axs[3].set_title('RMS Step Convergence')
    axs[3].set_ylabel('RMS Step')
    axs[3].grid(True)

    x_max_step = range(len(max_steps))
    axs[4].fill_between(x_max_step, -abs(max_step_tolerance[0]), abs(max_step_tolerance[0]), color='orange', alpha=0.2)
    axs[4].plot(max_steps, marker='o')
    axs[4].set_title('MAX Step Convergence')
    axs[4].set_ylabel('MAX Step')
    axs[4].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    if len(sys.argv) == 2:  # Only the orca output file is provided
        p = sys.argv[1]
        plot_convergence(p)
        plt.show()
    elif len(sys.argv) == 3:  # Both the orca output file and save path are provided
        p = sys.argv[1]
        save = sys.argv[2]
        plot_convergence(p)
        plt.savefig(save)
    else:
        print("Usage: geometry_convergence_plot <path_to_orca_out> [path_to_save_plot]")
        sys.exit(1)

if __name__ == "__main__":
    main()
