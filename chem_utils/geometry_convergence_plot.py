import matplotlib.pyplot as plt
import re
import sys
import numpy as np


def extract_all_convergence_data(text_content):
    # Initialize an empty dictionary to store extracted data
    data = {}
    all_criteria = set()  # Keep track of all criteria encountered

    # Pattern to match criterion, value, and tolerance
    pattern = re.compile(
        r"([A-Za-z\s\(\)\|]+)\s+(-?\d+\.\d+|\d+\.\d+e[+-]\d+)\s+(-?\d+\.\d+|\d+\.\d+e[+-]\d+)")

    # Flags
    is_convergence_block = False
    is_header_line = False
    block_data = {}  # Temporary storage for each block
    block_number = 0

    # Loop through lines of the text content
    for line in text_content.split('\n'):
        if "Geometry convergence" in line or "CI-NEB convergence" in line:
            is_convergence_block = True
            block_number += 1
            continue
        if is_convergence_block:
            if "Item                value" in line:
                is_header_line = True
                continue  # Skip the header line
            if is_header_line and ("------" in line or "......" in line):
                is_header_line = False
                continue  # Skip the line following the header
            matches = pattern.findall(line)
            for match_group in matches:
                criterion, value, tolerance = match_group
                criterion = criterion.strip()
                all_criteria.add(criterion)
                block_data[criterion] = {"values": float(
                    value), "tolerances": float(tolerance)}
            if "........................................................" in line or "-----" in line:
                is_convergence_block = False

                # Append data to the main data dictionary
                for criterion in all_criteria:
                    if criterion not in data:
                        data[criterion] = {"values": [
                            np.nan]*block_number, "tolerances": [np.nan]*block_number}
                    if criterion in block_data:
                        data[criterion]["values"].append(
                            block_data[criterion]["values"])
                        data[criterion]["tolerances"].append(
                            block_data[criterion]["tolerances"])
                    else:
                        data[criterion]["values"].append(np.nan)
                        data[criterion]["tolerances"].append(np.nan)

                block_data = {}

    # Before returning data, fill missing tolerances with nearest neighbor values
    for criterion, values_dict in data.items():
        values = values_dict["tolerances"]
        for i in range(len(values)):
            if np.isnan(values[i]):
                # Find the nearest non-nan value
                non_nan_values = [val for val in values if not np.isnan(val)]
                if i == 0:  # If the first value is nan
                    values[i] = non_nan_values[0]
                elif i == len(values) - 1:  # If the last value is nan
                    values[i] = non_nan_values[-1]
                else:
                    # Find the nearest non-nan value
                    left_non_nan = next(
                        (x for x in values[i::-1] if not np.isnan(x)), None)
                    right_non_nan = next(
                        (x for x in values[i:] if not np.isnan(x)), None)
                    if left_non_nan is None:
                        values[i] = right_non_nan
                    elif right_non_nan is None:
                        values[i] = left_non_nan
                    else:
                        # Choose the closer non-nan value
                        values[i] = left_non_nan if (
                            i - values[i::-1].index(left_non_nan)) < values[i:].index(right_non_nan) else right_non_nan

    return data


def plot_convergence(text_content, print_data=True):
    data = extract_all_convergence_data(text_content)
    # Print the extracted data if print_data is True
    if print_data:
        for criterion, values in data.items():
            print(f"{criterion}:")
            print("Values:", values["values"])
            print("Tolerances:", values["tolerances"])
            print("-" * 50)

    # Determine the number of subplots based on the data
    n_plots = len(data)

    # Plot
    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))

    if n_plots == 1:
        axs = [axs]

    idx = 0
    for criterion, values in data.items():
        x = range(len(values["values"]))
        axs[idx].fill_between(x, [-abs(t) for t in values["tolerances"]], [abs(t)
                              for t in values["tolerances"]], color='blue', alpha=0.2, zorder=1)

        # Set color based on convergence region
        colors = ['red' if abs(val) > tol else 'blue' for val, tol in zip(
            values["values"], values["tolerances"])]
        axs[idx].scatter(x, values["values"], c=colors, zorder=3)
        axs[idx].plot(values["values"], color='black',
                      linestyle='dashed', zorder=2)

        axs[idx].set_title(f'{criterion} Convergence')
        axs[idx].set_ylabel(criterion)
        axs[idx].set_yscale('symlog')  # Set y-axis to symlog scale
        axs[idx].set_xticks(x)
        axs[idx].grid(True)

        # Adjust y-ticks to 2 decimal places
        locs = axs[idx].get_yticks()
        new_locs = [round(loc, 2) for loc in locs]
        axs[idx].set_yticks(new_locs)
        axs[idx].set_yticklabels([f"{loc:.2f}" for loc in new_locs])

        idx += 1

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) == 2:  # Only the orca output file is provided
        p = sys.argv[1]
        with open(p, 'r') as f:
            content = f.read()
        plot_convergence(content)
    elif len(sys.argv) == 3:  # Both the orca output file and save path are provided
        p = sys.argv[1]
        save = sys.argv[2]
        with open(p, 'r') as f:
            content = f.read()
        plot_convergence(content)
        plt.savefig(save)
    else:
        print(
            "Usage: geometry_convergence_plot <path_to_orca_out> [path_to_save_plot]")
        sys.exit(1)


if __name__ == "__main__":
    main()
