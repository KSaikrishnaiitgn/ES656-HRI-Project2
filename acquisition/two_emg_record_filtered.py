
import serial
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path

from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from scipy.ndimage import uniform_filter1d


# =====================================================
# 1. CONFIGURATION
# =====================================================

PORT = "COM10"
BAUD_RATE = 1000000

FS = 2000

ADC_REFERENCE = 3.3
ADC_MAX = 4095

RECORD_DURATION = 20

LOW_CUTOFF = 20
HIGH_CUTOFF = 450

NOTCH_FREQ = 50
NOTCH_Q = 30

WINDOW_MS = 100


# =====================================================
# 2. FILE NAMES
# =====================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

raw_filename = f"Walking_constraint_marker2_{timestamp}.csv"

filtered_filename = f"Walking_constraint_marker2_{timestamp}.csv"

plot_filename = f"Walking_constraint_marker2_{timestamp}.png"


# =====================================================
# 3. DATA STORAGE
# =====================================================

# Channel 1: Tibialis Anterior
tibialis_adc = []
tibialis_voltage = []

# Channel 2: Gastrocnemius Medialis
gastro_adc = []
gastro_voltage = []

host_times = []

sample_count = 0
ser = None

start_time = None
recording_error = None


# =====================================================
# 4. SERIAL RECORDING
# =====================================================

try:

    print("\n========================================")
    print("       DUAL CHANNEL EMG ACQUISITION")
    print("========================================")

    print(f"\nConnecting to {PORT}...")

    ser = serial.Serial(
        port=PORT,
        baudrate=BAUD_RATE,
        timeout=0.1
    )

    time.sleep(2)

    ser.reset_input_buffer()

    print("\nConnected successfully!")

    print("Channel 1: Tibialis Anterior (GPIO26)")
    print("Channel 2: Gastrocnemius Medialis (GPIO27)")

    print(f"Sampling frequency target: {FS} Hz/channel")
    print(f"Recording duration: {RECORD_DURATION} seconds")

    print("\nRecording started...")
    print("Press CTRL+C to stop manually.\n")

    start_time = time.perf_counter()

    next_report = 1

    with open(raw_filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Sample",
            "Host_Elapsed_Time_s",
            "Nominal_Sample_Time_s",
            "Tibialis_ADC",
            "Tibialis_Voltage_V",
            "Gastro_ADC",
            "Gastro_Voltage_V"
        ])

        while True:

            elapsed = time.perf_counter() - start_time

            # Automatic stop
            if RECORD_DURATION is not None:

                if elapsed >= RECORD_DURATION:
                    print("\nRecording duration completed.")
                    break

            # Read one simultaneous sample pair
            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:
                continue

            # Parse CSV serial data
            try:

                values = line.split(",")

                if len(values) != 2:
                    continue

                adc1 = int(values[0])
                adc2 = int(values[1])

            except ValueError:
                continue

            # Validate ADC readings
            if not (0 <= adc1 <= ADC_MAX):
                continue

            if not (0 <= adc2 <= ADC_MAX):
                continue

            # Convert ADC readings into voltage
            voltage1 = (
                adc1 / ADC_MAX
            ) * ADC_REFERENCE

            voltage2 = (
                adc2 / ADC_MAX
            ) * ADC_REFERENCE

            elapsed = time.perf_counter() - start_time

            nominal_time = sample_count / FS

            sample_count += 1

            # Store Channel 1
            tibialis_adc.append(adc1)
            tibialis_voltage.append(voltage1)

            # Store Channel 2
            gastro_adc.append(adc2)
            gastro_voltage.append(voltage2)

            host_times.append(elapsed)

            # Save raw data
            writer.writerow([
                sample_count,
                round(elapsed, 6),
                round(nominal_time, 6),
                adc1,
                round(voltage1, 6),
                adc2,
                round(voltage2, 6)
            ])

            # Flush periodically
            if sample_count % 500 == 0:
                file.flush()

            # Display progress once per second
            if elapsed >= next_report:

                print(
                    f"Time: {elapsed:.1f}s | "
                    f"Samples: {sample_count} | "
                    f"Tibialis: {adc1} | "
                    f"Gastro: {adc2}"
                )

                next_report += 1


except KeyboardInterrupt:

    print("\nRecording stopped manually.")


except serial.SerialException as e:

    recording_error = str(e)

    print(f"\nSerial communication error: {e}")


except Exception as e:

    recording_error = str(e)

    print(f"\nUnexpected error: {e}")


finally:

    if ser is not None:

        try:

            if ser.is_open:
                ser.close()

        except Exception:
            pass

    print("\nSerial connection closed.")


# =====================================================
# 5. FILTER DESIGN
# =====================================================

if sample_count > 1000:

    print("\nProcessing both EMG channels...")

    tibialis_voltage = np.asarray(tibialis_voltage)
    gastro_voltage = np.asarray(gastro_voltage)

    tibialis_adc = np.asarray(tibialis_adc)
    gastro_adc = np.asarray(gastro_adc)

    time_axis = np.arange(sample_count) / FS

    # Butterworth bandpass: 20-450 Hz
    sos = butter(
        4,
        [LOW_CUTOFF, HIGH_CUTOFF],
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    # 50 Hz notch
    b_notch, a_notch = iirnotch(
        NOTCH_FREQ,
        NOTCH_Q,
        fs=FS
    )


    # =================================================
    # 6. CHANNEL 1: TIBIALIS PROCESSING
    # =================================================

    print("Filtering Tibialis Anterior...")

    tibialis_bandpassed = sosfiltfilt(
        sos,
        tibialis_voltage
    )

    tibialis_filtered = filtfilt(
        b_notch,
        a_notch,
        tibialis_bandpassed
    )

    # Full wave rectification
    tibialis_rectified = np.abs(
        tibialis_filtered
    )

    # Sliding RMS
    window_samples = int(
        FS * WINDOW_MS / 1000
    )

    tibialis_squared = tibialis_filtered ** 2

    tibialis_moving_mean = uniform_filter1d(
        tibialis_squared,
        size=window_samples,
        mode="reflect"
    )

    tibialis_rms = np.sqrt(
        np.maximum(tibialis_moving_mean, 0)
    )


    # =================================================
    # 7. CHANNEL 2: GASTRO PROCESSING
    # =================================================

    print("Filtering Gastrocnemius Medialis...")

    gastro_bandpassed = sosfiltfilt(
        sos,
        gastro_voltage
    )

    gastro_filtered = filtfilt(
        b_notch,
        a_notch,
        gastro_bandpassed
    )

    # Full wave rectification
    gastro_rectified = np.abs(
        gastro_filtered
    )

    # Sliding RMS
    gastro_squared = gastro_filtered ** 2

    gastro_moving_mean = uniform_filter1d(
        gastro_squared,
        size=window_samples,
        mode="reflect"
    )

    gastro_rms = np.sqrt(
        np.maximum(gastro_moving_mean, 0)
    )


    # =================================================
    # 8. SAVE PROCESSED CSV
    # =================================================

    with open(filtered_filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Time_s",

            "Tibialis_Raw_V",
            "Tibialis_Filtered_V",
            "Tibialis_Rectified_V",
            "Tibialis_RMS_V",

            "Gastro_Raw_V",
            "Gastro_Filtered_V",
            "Gastro_Rectified_V",
            "Gastro_RMS_V"
        ])

        for i in range(sample_count):

            writer.writerow([
                round(time_axis[i], 6),

                round(tibialis_voltage[i], 6),
                round(tibialis_filtered[i], 6),
                round(tibialis_rectified[i], 6),
                round(tibialis_rms[i], 6),

                round(gastro_voltage[i], 6),
                round(gastro_filtered[i], 6),
                round(gastro_rectified[i], 6),
                round(gastro_rms[i], 6)
            ])

    print("\nBoth channels processed successfully!")


    # =================================================
    # 9. PLOT BOTH EMG CHANNELS
    # =================================================

    fig, axes = plt.subplots(
        4, 2,
        figsize=(16, 12),
        sharex=True
    )

    fig.suptitle(
        "DUAL CHANNEL EMG ANALYSIS",
        fontsize=16
    )

    # ---------------------------------------------
    # ROW 1: RAW SIGNALS
    # ---------------------------------------------

    axes[0, 0].plot(
        time_axis,
        tibialis_voltage,
        linewidth=0.5
    )

    axes[0, 0].set_title("1. TIBIALIS - RAW EMG")
    axes[0, 0].set_ylabel("Voltage (V)")
    axes[0, 0].grid(True)

    axes[0, 1].plot(
        time_axis,
        gastro_voltage,
        linewidth=0.5
    )

    axes[0, 1].set_title("1. GASTRO - RAW EMG")
    axes[0, 1].set_ylabel("Voltage (V)")
    axes[0, 1].grid(True)


    # ---------------------------------------------
    # ROW 2: FILTERED SIGNALS
    # ---------------------------------------------

    axes[1, 0].plot(
        time_axis,
        tibialis_filtered,
        linewidth=0.5
    )

    axes[1, 0].set_title("2. TIBIALIS - FILTERED EMG")
    axes[1, 0].set_ylabel("Voltage (V)")
    axes[1, 0].grid(True)

    axes[1, 1].plot(
        time_axis,
        gastro_filtered,
        linewidth=0.5
    )

    axes[1, 1].set_title("2. GASTRO - FILTERED EMG")
    axes[1, 1].set_ylabel("Voltage (V)")
    axes[1, 1].grid(True)


    # ---------------------------------------------
    # ROW 3: RECTIFIED SIGNALS
    # ---------------------------------------------

    axes[2, 0].plot(
        time_axis,
        tibialis_rectified,
        linewidth=0.5
    )

    axes[2, 0].set_title("3. TIBIALIS - RECTIFIED EMG")
    axes[2, 0].set_ylabel("Amplitude (V)")
    axes[2, 0].grid(True)

    axes[2, 1].plot(
        time_axis,
        gastro_rectified,
        linewidth=0.5
    )

    axes[2, 1].set_title("3. GASTRO - RECTIFIED EMG")
    axes[2, 1].set_ylabel("Amplitude (V)")
    axes[2, 1].grid(True)


    # ---------------------------------------------
    # ROW 4: RMS ENVELOPES
    # ---------------------------------------------

    axes[3, 0].plot(
        time_axis,
        tibialis_rms,
        linewidth=1.2
    )

    axes[3, 0].set_title(
        f"4. TIBIALIS - RMS ({WINDOW_MS} ms)"
    )

    axes[3, 0].set_xlabel("Time (seconds)")
    axes[3, 0].set_ylabel("RMS (V)")
    axes[3, 0].grid(True)

    axes[3, 1].plot(
        time_axis,
        gastro_rms,
        linewidth=1.2
    )

    axes[3, 1].set_title(
        f"4. GASTRO - RMS ({WINDOW_MS} ms)"
    )

    axes[3, 1].set_xlabel("Time (seconds)")
    axes[3, 1].set_ylabel("RMS (V)")
    axes[3, 1].grid(True)


    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(
        plot_filename,
        dpi=200
    )

    plt.show()

    print(f"\nRaw CSV: {Path(raw_filename).resolve()}")
    print(f"Processed CSV: {Path(filtered_filename).resolve()}")
    print(f"Graph: {Path(plot_filename).resolve()}")


else:

    print("\nInsufficient samples for processing.")


# =====================================================
# 10. SUMMARY
# =====================================================

print("\n========================================")
print("           DUAL EMG SUMMARY")
print("========================================")

print(f"Total simultaneous sample pairs: {sample_count}")

if len(host_times) > 1:

    received_rate = (
        (sample_count - 1) /
        (host_times[-1] - host_times[0])
    )

    print(f"Approx. received rate: {received_rate:.2f} Hz")

if sample_count > 1000:

    print("\nTibialis RMS mean:",
          f"{np.mean(tibialis_rms):.5f} V")

    print("Gastro RMS mean:",
          f"{np.mean(gastro_rms):.5f} V")

print(f"\nRaw data: {raw_filename}")
print(f"Processed data: {filtered_filename}")
print(f"Plot: {plot_filename}")

if recording_error:
    print(f"Recording error: {recording_error}")

print("\nAcquisition completed.")
print("========================================")
