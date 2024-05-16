import numpy as np
import matplotlib.pyplot as plt


def SimpleGraph(list_par, x_line=None, show=True):
    if not isinstance(list_par, list):
        raise TypeError(f'list_par is not list')
    if not x_line:
        x_line = range(len(list_par[0]))
    for index, par in enumerate(list_par):
        plt.plot(x_line, par, label=f"Line {index}")
    plt.grid()
    plt.legend()
    if show:
        plt.show()
    return plt


def DistributionGraph(list_data):
    plt.hist(list_data, bins='auto')
    plt.title('Histogram of Values')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.show()


def DistributionGroupsAnalyzer(data_column,
                               num_groups=5,
                               view=True,
                               nround=None):
    array = data_column.to_numpy()
    sorted_array = np.sort(array)
    num_elements = len(sorted_array)
    elements_per_group = num_elements // num_groups

    group_boundaries = []
    for i in range(num_groups):
        start_index = i * elements_per_group
        end_index = start_index + elements_per_group - 1
        group_boundaries.append((sorted_array[start_index], sorted_array[end_index]))

    if view:
        plt.hist(sorted_array, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Value')
        plt.ylabel('Freq')
        plt.title('DistributionGroups')

        boundaries_str = '\n'.join(
            [f'({start:.2f}, {end:.2f})' for i, (start, end) in enumerate(group_boundaries)])
        plt.text(0.7, 0.8, boundaries_str, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.5))

        plt.show()

    print(f"| DistributionGroups results (num_groups: {num_groups}):")
    if nround:
        print(f"| {np.round(group_boundaries, nround)}")
    else:
        print(f"| {group_boundaries}")

    return group_boundaries


def DistributionAnalyzer(data_column,
                         n_filter=3):

    value_counts = data_column.value_counts()
    filtered_value_counts = value_counts[value_counts >= n_filter]

    plt.hist(filtered_value_counts, bins=10, edgecolor='k')
    plt.title(f'Distribution')
    plt.ylabel('Frequency')
    plt.show()


def DrawTimeSeriesWithSmooth(data_column,
                             mask_smooth=None,
                             interactive=False,
                             index_fill=None):
    if mask_smooth is None:
        mask_smooth = [len(data_column) // it // 3 for it in range(1, 4)]

    if interactive:
        plt.clf()

    plt.plot(data_column, marker='o')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Graph')

    for ws in mask_smooth:
        window_size = ws
        smoothed_column = data_column.rolling(window=window_size, min_periods=1, center=True).mean()
        plt.plot(smoothed_column)

    if index_fill:
        plt.axvspan(index_fill[0], index_fill[1] + 1, color='gray', alpha=0.5)

    if interactive:
        plt.pause(0.01)
        plt.draw()
    else:
        plt.show()


def DrawPredictGradient(gradient_data,
                        source_color=0.6,
                        gain_max_num=False,
                        labels_horizontal=None,
                        labels_vertical=None,
                        interactive=False):
    if gain_max_num:
        gradient_data = gradient_data / np.max(gradient_data)
    width = len(gradient_data[0])
    if width // 2 == 1:
        mid_width = (width + 1) // 2
    else:
        mid_width = width // 2
    height, width = gradient_data.shape
    colors = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            red, green, blue, sat = source_color, source_color, source_color, gradient_data[y][x]
            if x < mid_width:
                red = red + (1.0 - source_color) * sat
            elif x > mid_width:
                green = green + (1.0 - source_color) * sat
            else:
                blue = blue + (1.0 - source_color) * sat
            color = np.array([red, green, blue])
            colors[y, x] = color / np.sum(color)
    if interactive:
        plt.clf()
    plt.imshow(colors, aspect='auto', vmin=0, vmax=255)
    if labels_horizontal:
        local_labels_horizontal = labels_horizontal.copy()
        local_labels_horizontal[0] = ('-inf', local_labels_horizontal[0])
        local_labels_horizontal[-1] = (local_labels_horizontal[-1], 'inf')
        local_labels_horizontal = [f"{el[0]}\n{el[-1]}" for el in local_labels_horizontal]
        plt.xticks(range(len(local_labels_horizontal)), local_labels_horizontal)
    if labels_vertical:
        local_labels_vertical = labels_vertical.copy()
        local_labels_vertical = [str(el) for el in local_labels_vertical]
        plt.yticks(range(len(local_labels_vertical)), local_labels_vertical)
    if interactive:
        plt.pause(0.01)
        plt.draw()
    else:
        plt.show()


def DrawLearningCurve(history, metric_name):

    metric_values = history.history[metric_name]
    epochs = range(1, len(metric_values) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metric_values, 'b', label=f'Training {metric_name}')
    plt.title(f'Training {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()


def ConsoleHistogram(data,
                     min_val=None,
                     max_val=None,
                     size=30,
                     label=None):
    if min_val is None:
        min_val = min(data)
    if max_val is None:
        max_val = max(data)
    h_data = [int((x / abs(max_val - min_val)) * size) for x in data]
    if label:
        transformed_label = [str(element).ljust(5) if len(str(element)) < 5 else str(element)[:5] for element in label]
        str_label = [f"|{element}" for element in transformed_label]
    else:
        str_label = ["" for _ in range(len(data))]
    print('=' * (size + 8 + len(str_label[0])))
    for i, val in enumerate(h_data):
        str_val = str(round(data[i], 3))
        print(str_label[i] + "|" + "#" * val + " " * (size - val) + "|" + str_val + "|")
    print('=' * (size + 8 + len(str_label[0])))


def MatrixGradient(matrix, turn_over=False):
    min_val = matrix.min()
    max_val = matrix.max()
    normalized_matrix = (matrix - min_val) / (max_val - min_val)

    if turn_over:
        normalized_matrix = np.rot90(normalized_matrix)

    plt.imshow(normalized_matrix, cmap='gray', vmin=0, vmax=1)
    plt.show()
