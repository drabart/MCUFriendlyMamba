import serial
import sys

# --- CONFIGURATION ---
# Change this to match your ESP32-S3 port (e.g., 'COM3' on Windows or '/dev/ttyACM0' on Linux/Mac)
SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 115200
OUTPUT_FILENAME = 'esp32_recording.wav'
# ---------------------

def capture_audio():
    print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        print("Make sure no other program (like Arduino IDE) is using the port.")
        return

    print("Listening for ESP32... Please start or reset your ESP32 recording.")
    
    recording = False
    hex_data = []

    while True:
        try:
            # Read a line from serial and decode safely
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if not line:
                continue

            if "Recording started" in line:
                print("Recording started")
                continue

            if "=== WAV FILE START ===" in line:
                print("Recording stream detected! Capturing data...")
                recording = True
                hex_data = []
                continue

            if "=== WAV FILE END ===" in line:
                if recording:
                    print("Finished receiving data. Processing file...")
                    break

            if recording:
                # Keep track of the hex string coming in
                hex_data.append(line)
                # Visual indicator so you know it hasn't frozen
                sys.stdout.write('.')
                sys.stdout.flush()
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            ser.close()
            return

    ser.close()

    # Join all chunks together
    full_hex_string = "".join(hex_data)
    
    # Strip any accidental whitespace or newlines embedded in the data
    full_hex_string = "".join(full_hex_string.split()) 

    try:
        # Convert hex string back to binary bytes
        audio_bytes = bytes.fromhex(full_hex_string)
        
        # Write to final WAV file
        with open(OUTPUT_FILENAME, 'wb') as wav_file:
            wav_file.write(audio_bytes)
            
        print(f"\nSuccess! Audio saved to: '{OUTPUT_FILENAME}'")
        print(f"Total file size: {len(audio_bytes)} bytes")
        
    except ValueError as ve:
        print("\nError: The captured data contained non-hexadecimal characters.")
        print("This usually happens if debug logs (ESP_LOGI) printed out mid-stream.")

if __name__ == "__main__":
    capture_audio()