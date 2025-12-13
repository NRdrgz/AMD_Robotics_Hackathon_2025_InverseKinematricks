#!/usr/bin/env python3
"""Detect Arduino serial port. Run this script to find the correct port."""

from serial.tools import list_ports


def find_arduino_port():
    """Find the Arduino serial port automatically."""
    # Common Arduino VID/PID combinations
    arduino_vid_pids = [
        (0x2341, 0x0043),  # Arduino Uno
        (0x2341, 0x0001),  # Arduino Uno
        (0x2A03, 0x0043),  # Arduino Uno (Chinese clone)
        (0x2341, 0x0010),  # Arduino Mega 2560
        (0x2341, 0x0042),  # Arduino Mega 2560
        (0x2A03, 0x0010),  # Arduino Mega 2560 (Chinese clone)
    ]

    print("Scanning for Arduino devices...\n")

    # First, try to find by VID/PID
    for port in list_ports.comports():
        if (port.vid, port.pid) in arduino_vid_pids:
            print(f"✓ Found Arduino at: {port.device}")
            print(f"  Description: {port.description}")
            print(f"  VID: 0x{port.vid:04X}, PID: 0x{port.pid:04X}")
            print(f"\nUse this port in your code: {port.device}")
            return port.device

    # Fallback: look for common port patterns
    print("No Arduino found by VID/PID. Checking common port patterns...\n")
    common_patterns = ["usbmodem", "ttyUSB", "ttyACM", "COM"]
    for port in list_ports.comports():
        port_lower = port.device.lower()
        if any(pattern in port_lower for pattern in common_patterns):
            print(f"⚠ Found potential serial device at: {port.device}")
            print(f"  Description: {port.description}")
            if port.vid and port.pid:
                print(f"  VID: 0x{port.vid:04X}, PID: 0x{port.pid:04X}")
            print(f"\nUse this port in your code: {port.device}")
            return port.device

    # If nothing found, list all available ports
    print("❌ No Arduino found. Available serial ports:\n")
    if not list(list_ports.comports()):
        print("  (no serial ports found)")
    else:
        for port in list_ports.comports():
            vid_pid = ""
            if port.vid and port.pid:
                vid_pid = f" (VID: 0x{port.vid:04X}, PID: 0x{port.pid:04X})"
            print(f"  - {port.device}: {port.description}{vid_pid}")

    return None


if __name__ == "__main__":
    port = find_arduino_port()
    if port:
        print(f"\n✅ Detection successful! Port: {port}")
    else:
        print("\n❌ Could not detect Arduino port.")
        print("   Make sure your Arduino is connected and try again.")
