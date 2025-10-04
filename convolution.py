import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.collections import LineCollection

def create_filter(signal, center_position, filter_length, samples_per_period, filter_type="normal"):
    """
    Create a filter from the signal at any specified position.
    
    Args:
        signal: The input signal array
        center_position: Position in the first period (0 to 2π) as a fraction (0.0 to 1.0)
        filter_length: Length of the filter in samples
        samples_per_period: Number of samples per period
        filter_type: Type of filter to create ("normal" or "zero_crossing")
    """
    # Convert the position fraction to an index in the first period
    center_index = int(center_position * samples_per_period)
    filter_start = center_index - filter_length // 2
    filter_end = filter_start + filter_length
    
    # Ensure we don't go out of bounds for the first period
    filter_start = max(0, filter_start)
    filter_end = min(samples_per_period, filter_end)
    
    if filter_type == "zero_crossing":
        # Create a specialized zero crossing filter
        # This captures the transition through zero
        t = np.linspace(-np.pi/2, np.pi/2, filter_length)
        return np.sin(t), filter_start
    else:
        return signal[filter_start:filter_end], filter_start

def find_zero_crossings(signal):
    """Find the rising zero crossing points in a signal (low to high transitions)."""
    # Find where the signal changes sign from negative to positive
    zero_crossings = np.where((signal[:-1] < 0) & (signal[1:] > 0))[0]
    
    # Interpolate to get more precise zero crossing points
    precise_crossings = []
    for zc in zero_crossings:
        if zc < len(signal) - 1:  # Ensure we're not at the last point
            # Linear interpolation between points
            x0, x1 = zc, zc + 1
            y0, y1 = signal[zc], signal[zc + 1]
            if y0 != y1:  # Avoid division by zero
                precise_x = x0 - y0 * (x1 - x0) / (y1 - y0)
                precise_crossings.append(precise_x)
    return np.array(precise_crossings)

def analyze_filter(signal, filter_vector, samples_per_period, threshold=0.5, filter_type="normal", name=""):
    """
    Analyze how well a filter detects patterns in the signal.
    
    Args:
        signal: The input signal array
        filter_vector: The filter to use for detection
        samples_per_period: Number of samples per period
        threshold: Minimum peak height relative to maximum
        filter_type: Type of filter being used
        name: Name of the filter for adaptive thresholding
    """
    # Perform convolution
    conv = np.convolve(signal, filter_vector, mode='same')
    
    # Normalize convolution to range [-1, 1]
    conv = conv / np.max(np.abs(conv))
    
    # Initialize properties
    properties = {}
    
    # Adjust threshold based on filter type and signal region
    if "Long Peak" in name:
        # Lower threshold for long peak filter due to broader correlation peaks
        base_threshold = 0.3
        
        # Adaptive threshold for noisy regions
        # Use a rolling window to detect high-variance regions
        window_size = samples_per_period // 4
        rolling_std = np.array([np.std(signal[max(0, i-window_size):min(len(signal), i+window_size)]) 
                              for i in range(len(signal))])
        
        # Create an adaptive threshold based on local signal variance
        is_noisy = rolling_std > np.median(rolling_std) * 1.5
        adaptive_threshold = np.where(is_noisy, base_threshold * 0.7, base_threshold)
        
        # Use minimum distance of 60% of period in noisy regions to catch more peaks
        min_distance = int(samples_per_period * (0.6 if np.any(is_noisy) else 0.8))
        
        # Store noise information for visualization
        properties.update({
            "is_noisy": is_noisy,
            "rolling_std": rolling_std,
            "adaptive_threshold": adaptive_threshold
        })
        
        threshold = base_threshold
    elif "Falling Slope" in name:
        # More sensitive parameters for falling slope detection
        threshold = 0.2  # Lower threshold to catch slopes
        min_distance = int(samples_per_period * 0.8)  # Minimum distance between detections
        
        # Calculate slope of signal for better detection
        signal_slope = np.gradient(signal)
        slope_conv = np.convolve(signal_slope, filter_vector, mode='same')
        slope_conv = slope_conv / np.max(np.abs(slope_conv))
        
        # Update convolution to focus on negative slopes
        conv = -slope_conv  # Negative slopes will now appear as positive peaks
        
        properties.update({
            "slope_based": True
        })
    else:
        min_distance = samples_per_period//2
    
    if filter_type == "zero_crossing":
        # For zero crossings, use a lower threshold and smaller minimum distance
        threshold = 0.2  # Lower threshold for zero crossings
        min_distance = samples_per_period // 2  # Distance between rising crossings
        
        # Find only positive peaks (corresponding to rising zero crossings)
        peaks, properties_pos = find_peaks(conv, height=threshold, distance=min_distance,
                                         prominence=0.1)  # Add prominence requirement
        
        # Get actual zero crossings for comparison
        true_crossings = find_zero_crossings(signal)
        
        # Calculate detection accuracy
        max_distance = samples_per_period // 8  # Max distance for matching peaks to crossings
        detected_crossings = []
        missed_crossings = []
        
        for tc in true_crossings:
            # Find closest peak to this crossing
            if len(peaks) > 0:
                min_dist = np.min(np.abs(peaks - tc))
                if min_dist < max_distance:
                    detected_crossings.append(tc)
                else:
                    missed_crossings.append(tc)
            else:
                missed_crossings.append(tc)
        
        detection_rate = len(detected_crossings) / len(true_crossings) if len(true_crossings) > 0 else 0
        
        properties = {
            "true_crossings": true_crossings,
            "detected_crossings": np.array(detected_crossings),
            "missed_crossings": np.array(missed_crossings),
            "detection_rate": detection_rate,
            "threshold": threshold,
            "min_distance": min_distance
        }
    else:
        # For other patterns, look for positive peaks
        peaks, properties = find_peaks(conv, height=threshold, distance=min_distance)
    
    # Add debug information
    properties = {} if properties is None else properties
    properties.update({
        "threshold": threshold,
        "min_distance": min_distance,
        "conv_max": np.max(conv),
        "conv_min": np.min(conv)
    })
    
    return conv, peaks, properties

def create_composite_signal(num_periods, samples_per_period):
    """
    Create a signal with different patterns for testing pattern detection.
    """
    total_samples = num_periods * samples_per_period
    t = np.linspace(0, num_periods * 2 * np.pi, total_samples)
    
    # Base sine wave
    signal = np.sin(t)
    
    # Add some interesting features:
    # 1. Double the amplitude in some regions
    signal[samples_per_period*2:samples_per_period*3] *= 2.0
    
    # 2. Add a square pulse in another region
    pulse_start = int(3.5 * samples_per_period)
    pulse_width = samples_per_period // 4
    signal[pulse_start:pulse_start+pulse_width] = 1.0
    
    # 3. Add some noise in another region
    noise_start = int(7.5 * samples_per_period)
    noise_end = int(8.5 * samples_per_period)
    signal[noise_start:noise_end] += np.random.normal(0, 0.2, noise_end-noise_start)
    
    return signal

def add_annotation_arrow(ax, x, y, text, dx=30, dy=20, connectionstyle="arc3,rad=0.2"):
    """Helper function to add an annotation arrow with consistent styling."""
    return ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy),
                      fontsize=8, color='#444444',
                      arrowprops=dict(
                          arrowstyle='->',
                          connectionstyle=connectionstyle,
                          color='#666666',
                          alpha=0.6
                      ),
                      bbox=dict(
                          boxstyle='round,pad=0.5',
                          fc='white',
                          ec='#666666',
                          alpha=0.8
                      ),
                      zorder=5)

def main():
    # 1. Generate signal vector with interesting patterns
    num_periods = 10
    samples_per_period = 200
    total_samples = num_periods * samples_per_period

    # Create signal with various patterns
    signal = create_composite_signal(num_periods, samples_per_period)

    # 2. Create filters at different positions and lengths
    # Position reference:
    # Order matters: filters will be displayed in this order from top to bottom
    filter_positions = [
        (0.0, 30, "Zero Crossing", "zero_crossing"),    # First plot after signal
        (0.33, 20, "Falling Slope", "normal"),         # Second plot
        (0.25, 40, "Long Peak Filter", "normal"),      # Third plot
    ]
    
    filters = []
    for pos, length, name, ftype in filter_positions:
        filt, start = create_filter(signal, pos, length, samples_per_period, filter_type=ftype)
        filters.append((filt, start, name, length, ftype))
    
    # Set style parameters
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.facecolor'] = '#f8f8f8'
    colors = plt.cm.Set2(np.linspace(0, 1, len(filter_positions)))
    
    # Create figure with subplots
    num_filters = len(filters)
    fig = plt.figure(figsize=(15, 2 + 4 + 3*num_filters))  # Added height for filter bank
    gs = plt.GridSpec(3 + num_filters, 1, height_ratios=[1.5, 2] + [1]*num_filters + [2])
    
    # Create filter bank subplot
    ax_filters = fig.add_subplot(gs[0])
    ax_filters.set_title('Filter Bank', fontsize=10, pad=10, fontweight='bold')
    ax_filters.axhline(y=0, color='#666666', linestyle='-', linewidth=0.5, zorder=1)
    
    # Plot filters side by side
    filter_spacing = 50
    current_pos = 0
    for i, (filt, start, name, length, ftype) in enumerate(filters):
        # Plot filter with enhanced styling
        filter_x = np.arange(len(filt)) + current_pos
        ax_filters.plot(filter_x, filt, color=colors[i], linewidth=2.5, 
                      label=name, solid_capstyle='round', zorder=4)
        # Add a light background for each filter
        ax_filters.fill_between(filter_x, filt, alpha=0.2, 
                             color=colors[i], zorder=3)
        current_pos += len(filt) + filter_spacing
    
    ax_filters.grid(True, alpha=0.3)
    ax_filters.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax_filters.set_ylabel('Amplitude')
    ax_filters.set_xlabel('Filter Samples')
    
    # Plot original signal
    ax_signal = fig.add_subplot(gs[1])
    ax_signal.plot(range(len(signal)), signal, color='#2F4F4F', linewidth=2, 
                  label='Signal', zorder=3)
    
    # Pre-compute all convolutions
    all_convolutions = []
    for filt, _, name, _, ftype in filters:
        threshold = 0.3 if ftype == "zero_crossing" else 0.5
        conv, peaks, properties = analyze_filter(signal, filt, samples_per_period, 
                                              threshold=threshold, filter_type=ftype, name=name)
        all_convolutions.append((conv, peaks, properties, name))
    
    # Add horizontal line at y=0
    ax_signal.axhline(y=0, color='#666666', linestyle='-', linewidth=0.5, zorder=1)
    
    # Add zero crossing markers on the original signal with enhanced visibility
    zero_crossings = find_zero_crossings(signal)
    for i, zc in enumerate(zero_crossings):
        # Add vertical lines
        ax_signal.vlines(zc, -1.5, 1.5, colors='#FFD700', linestyles='--', 
                        alpha=0.3, linewidth=2, zorder=2)
        # Add small marker at the zero crossing point
        ax_signal.plot(zc, 0, 'o', color='#FFD700', markersize=4, alpha=0.6, zorder=4)
        
        # Add annotations for different types of zero crossings
        if i < len(zero_crossings) - 1:
            if signal[int(zc) + 1] > 0:  # Rising zero crossing
                if i == 0:  # Annotate only the first one
                    add_annotation_arrow(ax_signal, zc, 0, "Rising\nZero Crossing",
                                      dx=30, dy=20)
            else:  # Falling zero crossing
                if i == 1:  # Annotate only the first one
                    add_annotation_arrow(ax_signal, zc, 0, "Falling\nZero Crossing",
                                      dx=-30, dy=20, connectionstyle="arc3,rad=-0.2")
    
    # Calculate max filter length for reference
    max_length = max(f[3] for f in filters)
    
    ax_signal.grid(True, alpha=0.3)
    ax_signal.legend(loc='upper right')
    ax_signal.set_xlabel('Sample Index')
    ax_signal.set_ylabel('Amplitude')
    ax_signal.set_title('Original Signal')
    ax_signal.set_xlim(-5, total_samples + 5)

    # Analyze and plot each filter's results
    detection_results = []
    for i, (filt, _, name, length, ftype) in enumerate(filters):
        # Analyze filter performance
        threshold = 0.3 if ftype == "zero_crossing" else 0.5  # Lower threshold for zero crossings
        conv, peaks, properties = analyze_filter(signal, filt, samples_per_period, 
                                              threshold=threshold, filter_type=ftype, name=name)
        
        # Create subplot for this filter's convolution
        ax = fig.add_subplot(gs[i+1])
        
        # Plot all convolutions with lower alpha
        for j, (other_conv, other_peaks, _, other_name) in enumerate(all_convolutions):
            if j != i:  # Plot other convolutions with medium alpha
                ax.plot(range(len(other_conv)), other_conv, color=colors[j], 
                       linewidth=1, alpha=0.5, label=f'{other_name}')
                if len(other_peaks) > 0:
                    ax.plot(other_peaks, other_conv[other_peaks], 'o', 
                           color=colors[j], markersize=4, alpha=0.5,
                           markeredgecolor='white', markeredgewidth=0.5)
        
        # Plot current convolution with full alpha
        ax.plot(range(len(conv)), conv, color=colors[i], linewidth=2, 
               label=f'{name} (current)', zorder=3)
               
        # Set y-axis limits to be symmetric around zero
        max_abs_val = max(abs(np.min([c[0] for c in all_convolutions])), 
                         abs(np.max([c[0] for c in all_convolutions])))
        ax.set_ylim(-max_abs_val * 1.1, max_abs_val * 1.1)  # Add 10% padding
        
        # Plot convolution result
        ax.plot(range(len(conv)), conv, color=colors[i], linewidth=1, alpha=0.7, label=f'{name} Conv.')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='#666666', linestyle='-', linewidth=0.5, zorder=1)
        
        # Plot convolution result with gradient
        points = np.array([conv, range(len(conv))]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(-1, 1)
        lc = LineCollection(segments, cmap='RdYlBu_r', norm=norm, zorder=2)
        lc.set_array(conv)
        line = ax.add_collection(lc)
        
        # Plot detected peaks with enhanced styling
        if len(peaks) > 0:
            # Add vertical highlight for peaks
            ax.vlines(peaks, [0], conv[peaks], colors=colors[i], 
                     linestyles='-', alpha=0.3, zorder=3)
            # Add peak markers
            ax.plot(peaks, conv[peaks], 'o', color=colors[i], 
                   markersize=8, label='Matches', zorder=4,
                   markeredgecolor='white', markeredgewidth=1)
        
        # For zero crossing filter, add additional visualization
        if ftype == "zero_crossing" and properties and "true_crossings" in properties:
            true_crossings = properties["true_crossings"]
            # Calculate detection accuracy
            max_distance = samples_per_period // 4
            matches = 0
            matched_crossings = []
            unmatched_crossings = []
            
            for tc in true_crossings:
                min_dist = np.min(np.abs(peaks - tc)) if len(peaks) > 0 else float('inf')
                if min_dist < max_distance:
                    matches += 1
                    matched_crossings.append(tc)
                else:
                    unmatched_crossings.append(tc)
            
            accuracy = matches / len(true_crossings) if len(true_crossings) > 0 else 0
            
            # Plot matched and unmatched crossings differently
            detected = properties.get("detected_crossings", [])
            missed = properties.get("missed_crossings", [])
            
            if len(detected) > 0:
                ax.vlines(detected, ax.get_ylim()[0], ax.get_ylim()[1],
                         colors='#90EE90', linestyles='--', alpha=0.3, 
                         label='Detected Crossings', zorder=2)
            if len(missed) > 0:
                ax.vlines(missed, ax.get_ylim()[0], ax.get_ylim()[1],
                         colors='#FFB6C1', linestyles='--', alpha=0.3,
                         label='Missed Crossings', zorder=2)
            
            # Add detection rate information
            detection_rate = properties.get("detection_rate", 0)
            ax.text(0.02, 0.98, 
                   f"Detection Parameters:\n"
                   f"Threshold: {properties['threshold']:.2f}\n"
                   f"Min Distance: {properties['min_distance']} samples\n"
                   f"Detection Rate: {detection_rate:.1%}",
                   transform=ax.transAxes, fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.8,
                           edgecolor='#666666', boxstyle='round,pad=0.5'),
                   verticalalignment='top')
            
            # Add accuracy indicator with color coding
            accuracy_color = plt.cm.RdYlGn(accuracy)
            ax.text(0.02, 0.98, f'Accuracy: {accuracy:.1%}',
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(facecolor=accuracy_color, alpha=0.3, edgecolor='none'),
                   verticalalignment='top')
            
            # Add detection pattern annotations
            if len(matched_crossings) > 0:
                # Annotate a good match
                good_match = matched_crossings[0]
                nearest_peak = peaks[np.argmin(np.abs(peaks - good_match))]
                midpoint_y = (conv[nearest_peak] + 0) / 2
                add_annotation_arrow(ax, good_match, midpoint_y,
                                  "Strong Match:\nHigh correlation at\nzero crossing",
                                  dx=40, dy=20)
            
            if len(unmatched_crossings) > 0:
                # Annotate a missed detection
                missed = unmatched_crossings[0]
                add_annotation_arrow(ax, missed, 0,
                                  "Missed Detection:\nWeak correlation",
                                  dx=-40, dy=-20,
                                  connectionstyle="arc3,rad=-0.2")
        
        # Add detection parameters for long peak filter
        if "Long Peak" in name:
            # Highlight noisy regions if detected
            if "is_noisy" in properties:
                noisy_regions = properties["is_noisy"]
                # Create spans for noisy regions
                in_noisy = False
                start_idx = 0
                for i, is_noisy in enumerate(noisy_regions):
                    if is_noisy and not in_noisy:
                        start_idx = i
                        in_noisy = True
                    elif not is_noisy and in_noisy:
                        ax.axvspan(start_idx, i, color='red', alpha=0.1, zorder=1)
                        in_noisy = False
                if in_noisy:  # If we end in a noisy region
                    ax.axvspan(start_idx, len(noisy_regions), color='red', alpha=0.1, zorder=1)
            
            info_text = (f"Detection Parameters:\n"
                        f"Base Threshold: {properties['threshold']:.2f}\n"
                        f"Min Distance: {properties['min_distance']} samples\n"
                        f"Max Conv: {properties['conv_max']:.2f}\n"
                        f"Min Conv: {properties['conv_min']:.2f}\n"
                        f"Adaptive: Lower threshold in noisy regions")
            ax.text(0.02, 0.98, info_text,
                   transform=ax.transAxes, fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.8,
                           edgecolor='#666666', boxstyle='round,pad=0.5'),
                   verticalalignment='top')
            
            # Add explanation of convolution values
            ax.text(0.98, 0.98,
                   "Convolution Values:\n" +
                   "→ Higher peaks indicate better pattern matches\n" +
                   "→ Both positive and negative peaks are significant\n" +
                   "→ Width indicates detection precision",
                   transform=ax.transAxes, fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.8,
                           edgecolor='#666666', boxstyle='round,pad=0.5'),
                   verticalalignment='top', horizontalalignment='right')
        
        # Enhance grid and labels
        ax.grid(True, alpha=0.2, zorder=1)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        ax.set_ylabel('Convolution Value', fontsize=9)
        ax.set_xlabel('Sample Index', fontsize=9)
        ax.set_xlim(0, total_samples)
        
        # Add title with enhanced styling
        if ftype == "zero_crossing":
            ax.set_title(f'Zero Crossing Detection Analysis', 
                        fontsize=10, pad=10, fontweight='bold')
        
        # Store detection results
        detection_results.append((name, len(peaks), length))

    # Create summary subplot
    ax_summary = fig.add_subplot(gs[-1])
    
    # Plot bar chart of detections
    x = np.arange(len(detection_results))
    ax_summary.bar(x, [r[1] for r in detection_results], color=colors)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels([r[0] for r in detection_results], rotation=45)
    ax_summary.set_ylabel('Number of Detections')
    ax_summary.set_title('Pattern Detection Summary')
    ax_summary.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save the figure
    plt.savefig('figures/pattern_detection_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print detection information
    print("\nDetection Results:")
    print(f"Signal length: {len(signal)} samples")
    print("\nFilter Performance:")
    for name, num_peaks, length in detection_results:
        print(f"{name} (length {length}): {num_peaks} matches found")

if __name__ == "__main__":
    main()