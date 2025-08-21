import numpy as np
from math import *
import calendar
import csv
import datetime
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import cv2
import os

@dataclass
class SolarPosition:
    """Class to store solar position data"""
    x: int
    y: int
    azimuth: float
    zenith: float
    mask: np.ndarray

class SunPositionVisualizerDemo:
    def __init__(self, 
                 latitude: float = 3.039201080912596,
                 longitude: float = 101.79437355320626,
                 time_zone_longitude: float = 120,
                 north_offset: float = -92.964,
                 image_radius: int = None,
                 image_center: Tuple[int, int] = None,
                 sun_mask_radius_ratio: float = 0.07,
                 image_size: Tuple[int, int] = None,
                 output_dir: str = "D:\\Image\\test_result\\sun_position_demo"):
        """Initialize the sun position calculator with visualization capabilities"""
        self.latitude = latitude
        self.longitude = longitude
        self.time_zone_longitude = time_zone_longitude
        self.north_offset = north_offset
        self.image_size = image_size or (1158, 1172)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate image properties
        height, width = self.image_size
        self.image_center = image_center or (width // 2, height // 2)
        self.image_radius = image_radius or min(width, height) // 2 - 10
        self.sun_mask_radius = max(7, int(self.image_radius * sun_mask_radius_ratio))

    def visualize_time_conversion(self, dt: datetime.datetime) -> Tuple[int, int]:
        """Visualize time conversion process"""
        print("\nStep 1: Time Conversion")
        print("-" * 50)
        
        # Calculate time correction
        correction = abs(60/15 * (self.longitude - self.time_zone_longitude))
        min_correction = int(correction)
        sec_correction = int((correction - min_correction) * 60)
        
        # Calculate time of day
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        correction_seconds = min_correction * 60 + sec_correction
        time_of_day = max(0, seconds - correction_seconds)
        
        # Calculate day of year
        months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if calendar.isleap(dt.year):
            months[1] = 29
        day_of_year = sum(months[:dt.month-1]) + dt.day
        
        # Visualize time conversion details
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        info_text = (
            f"Input DateTime: {dt}\n"
            f"Day of Year: {day_of_year}\n"
            f"Time of Day (seconds): {time_of_day}\n"
            f"Time Zone Correction: {min_correction}m {sec_correction}s\n"
            f"Longitude Difference: {abs(self.longitude - self.time_zone_longitude):.2f}°"
        )
        ax.text(0.1, 0.5, info_text, fontsize=12, va='center')
        plt.title("Time Conversion Process")
        plt.savefig(os.path.join(self.output_dir, "01_time_conversion.png"))
        plt.close()
        
        return day_of_year, time_of_day

    def visualize_solar_angles(self, day_of_year: int, time_of_day: int) -> Tuple[float, float]:
        """Visualize solar angle calculations"""
        print("\nStep 2: Solar Angle Calculations")
        print("-" * 50)
        
        # Calculate solar angles
        alpha = 2 * pi * (time_of_day - 43200) / 86400
        delta = radians(23.44 * sin(radians((360/365.25) * (day_of_year-80))))
        
        # Calculate zenith
        latitude_rad = radians(self.latitude)
        chi = acos(sin(delta) * sin(latitude_rad) + 
                  cos(delta) * cos(latitude_rad) * cos(alpha))
        
        # Calculate azimuth
        tan_xi = sin(alpha) / (sin(latitude_rad) * cos(alpha) - 
                              cos(latitude_rad) * tan(delta))
        
        # Determine azimuth quadrant
        if alpha > 0:
            xi = pi + atan(tan_xi) if tan_xi > 0 else 2*pi + atan(tan_xi)
        else:
            xi = atan(tan_xi) if tan_xi > 0 else pi + atan(tan_xi)
        
        azimuth = degrees(xi)
        zenith = degrees(chi)
        
        # Visualize angle calculations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Solar Position Diagram
        circle = plt.Circle((0.5, 0.5), 0.4, fill=False)
        ax1.add_artist(circle)
        
        # Add zenith line
        zenith_x = 0.5 + 0.4 * sin(chi)
        zenith_y = 0.5 + 0.4 * cos(chi)
        ax1.plot([0.5, zenith_x], [0.5, zenith_y], 'r-', label='Zenith Angle')
        
        # Add azimuth arc
        azimuth_rad = radians(azimuth)
        ax1.plot([0.5, 0.5 + 0.2 * cos(azimuth_rad)],
                 [0.5, 0.5 + 0.2 * sin(azimuth_rad)], 'b-', label='Azimuth')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.set_title('Solar Position Diagram')
        ax1.axis('equal')
        
        # Plot 2: Angle Information
        ax2.axis('off')
        info_text = (
            f"Intermediate Calculations:\n"
            f"α (hour angle): {degrees(alpha):.2f}°\n"
            f"δ (declination): {degrees(delta):.2f}°\n\n"
            f"Final Angles:\n"
            f"Azimuth: {azimuth:.2f}°\n"
            f"Zenith: {zenith:.2f}°"
        )
        ax2.text(0.1, 0.5, info_text, fontsize=12, va='center')
        
        plt.savefig(os.path.join(self.output_dir, "02_solar_angles.png"))
        plt.close()
        
        return azimuth, zenith

    def visualize_image_projection(self, azimuth: float, zenith: float, 
                                 image_path: str) -> SolarPosition:
        """Visualize the projection of solar position onto the image"""
        print("\nStep 3: Image Projection")
        print("-" * 50)
        
        # Load the image if provided
        if image_path and os.path.exists(image_path):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((*self.image_size, 3), dtype=np.uint8)
        
        # Calculate polar coordinates
        rho = zenith/90 * self.image_radius
        theta = azimuth - self.north_offset
        
        # Convert to Cartesian coordinates
        theta_rad = radians(theta)
        dx = rho * cos(theta_rad)
        dy = -rho * sin(theta_rad)
        
        sun_x = round(self.image_center[0] + 0.90 * dx)
        sun_y = round(self.image_center[1] + 0.90 * dy)
        
        # Ensure sun position is within bounds
        sun_x = max(self.sun_mask_radius, min(sun_x, self.image_size[1] - self.sun_mask_radius))
        sun_y = max(self.sun_mask_radius, min(sun_y, self.image_size[0] - self.sun_mask_radius))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Coordinate System
        ax1.plot([self.image_center[0], sun_x], [self.image_center[1], sun_y], 'r-', label='Position Vector')
        circle = plt.Circle(self.image_center, self.image_radius, fill=False, color='blue')
        ax1.add_artist(circle)
        ax1.plot(sun_x, sun_y, 'yo', label='Sun Position')
        
        # Draw coordinate axes
        ax1.axhline(y=self.image_center[1], color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=self.image_center[0], color='gray', linestyle='--', alpha=0.5)
        
        ax1.set_xlim(0, self.image_size[1])
        ax1.set_ylim(self.image_size[0], 0)  # Invert y-axis for image coordinates
        ax1.legend()
        ax1.set_title('Coordinate System')
        
        # Plot 2: Position Information
        ax2.axis('off')
        info_text = (
            f"Polar Coordinates:\n"
            f"ρ (rho): {rho:.2f} pixels\n"
            f"θ (theta): {theta:.2f}°\n\n"
            f"Image Coordinates:\n"
            f"x: {sun_x}\n"
            f"y: {sun_y}\n\n"
            f"Image Properties:\n"
            f"Center: {self.image_center}\n"
            f"Radius: {self.image_radius} pixels"
        )
        ax2.text(0.1, 0.5, info_text, fontsize=12, va='center')
        
        plt.savefig(os.path.join(self.output_dir, "03_image_projection.png"))
        plt.close()
        
        # Create and visualize sun mask
        mask = np.zeros(self.image_size, dtype=np.uint8)
        y, x = np.ogrid[:self.image_size[0], :self.image_size[1]]
        
        # Create circular mask for sun area
        sun_area = (x - sun_x)**2 + (y - sun_y)**2 <= self.sun_mask_radius**2
        fisheye_area = (x - self.image_center[0])**2 + (y - self.image_center[1])**2 <= self.image_radius**2
        mask[sun_area & fisheye_area] = 255
        
        # Visualize final result on image
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img)
        
        # Draw fisheye circle
        circle = plt.Circle(self.image_center, self.image_radius, fill=False, color='blue', alpha=0.5)
        ax.add_artist(circle)
        
        # Draw sun position and mask
        sun_circle = plt.Circle((sun_x, sun_y), self.sun_mask_radius, color='yellow', alpha=0.5)
        ax.add_artist(sun_circle)
        ax.plot(sun_x, sun_y, 'r+', markersize=10, label='Sun Center')
        
        ax.set_title('Final Sun Position on Image')
        ax.legend()
        plt.savefig(os.path.join(self.output_dir, "04_final_result.png"))
        plt.close()
        
        return SolarPosition(sun_x, sun_y, azimuth, zenith, mask)

    def calculate_position_demo(self, dt: datetime.datetime, image_path: str = None) -> SolarPosition:
        """Main demo function to calculate and visualize sun position"""
        print(f"\nStarting Sun Position Calculation Demo")
        print(f"{'='*50}")
        print(f"Input Parameters:")
        print(f"Date/Time: {dt}")
        print(f"Latitude: {self.latitude}°")
        print(f"Longitude: {self.longitude}°")
        print(f"North Offset: {self.north_offset}°")
        
        # Step 1: Time Conversion
        day_of_year, time_of_day = self.visualize_time_conversion(dt)
        
        # Step 2: Solar Angles
        azimuth, zenith = self.visualize_solar_angles(day_of_year, time_of_day)
        
        # Step 3: Image Projection
        position = self.visualize_image_projection(azimuth, zenith, image_path)
        
        print("\nDemo Completed!")
        print(f"Output files saved in: {self.output_dir}")
        print("Generated files:")
        for i, name in enumerate(['time_conversion', 'solar_angles', 
                                'image_projection', 'final_result'], 1):
            print(f"{i}. {i:02d}_{name}.png")
        
        return position

def main():
    # Test parameters
    image_path = "D:\\Image\\20250101_090400.jpg"
    output_dir = "D:\\Image\\test_result\\sun_position_demo"
    test_date = datetime.datetime(2025, 1, 1, 17, 4, 00)
    
    # Create calculator and run demo
    calculator = SunPositionVisualizerDemo(output_dir=output_dir)
    position = calculator.calculate_position_demo(test_date, image_path)
    
    print("\nFinal Results:")
    print(f"Sun position (x, y): ({position.x}, {position.y})")
    print(f"Solar angles (azimuth, zenith): ({position.azimuth:.2f}°, {position.zenith:.2f}°)")

if __name__ == "__main__":
    main() 