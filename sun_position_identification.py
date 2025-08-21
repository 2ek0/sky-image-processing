# -*- coding: utf-8 -*-
"""
Created on Aug 6 16:41:00 2019
Revised version on Feb 12 17:26:00 2020
@author: ynie
"""

import numpy as np
from math import *
import calendar
import csv
import datetime
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class SolarPosition:
    """Class to store solar position data"""
    x: int
    y: int
    azimuth: float
    zenith: float
    mask: np.ndarray

class TimeConverter:
    def __init__(self, longitude: float, time_zone_longitude: float):
        self.longitude = longitude
        self.time_zone_longitude = time_zone_longitude

    def convert_datetime(self, dt: datetime.datetime) -> Tuple[int, int]:
        """Convert datetime to day of year and time of day"""
        time_of_day = self._calculate_time_of_day(dt)
        day_of_year = self._calculate_day_of_year(dt)
        
        # if self._is_in_dst(dt, day_of_year):
        #     time_of_day -= 3600
            
        return day_of_year, time_of_day

    def _calculate_time_correction(self) -> Tuple[int, int]:
        """Calculate time correction in minutes and seconds"""
        correction = abs(60/15 * (self.longitude - self.time_zone_longitude))
        min_correction = int(correction)
        sec_correction = int((correction - min_correction) * 60)
        return min_correction, sec_correction

    def _calculate_time_of_day(self, dt: datetime.datetime) -> int:
        """Calculate time of day in seconds"""
        # Total seconds since midnight
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        
        # Apply time zone correction
        min_corr, sec_corr = self._calculate_time_correction()
        correction_seconds = min_corr * 60 + sec_corr
        
        # Adjust time ensuring we don't go below 0
        return max(0, seconds - correction_seconds)

    def _calculate_day_of_year(self, dt: datetime.datetime) -> int:
        """Calculate day of year considering leap years"""
        months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if self._is_leap_year(dt.year):
            months[1] = 29
        return sum(months[:dt.month-1]) + dt.day

    def _is_leap_year(self, year: int) -> bool:
        """Check if year is leap year"""
        return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)

    # def _is_in_dst(self, dt: datetime.datetime, day_of_year: int) -> bool:
    #     """Check if date is in DST period"""
    #     months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    #     dst_start = sum(months[:2]) + calendar.monthcalendar(dt.year, 3)[1][6]  # 2nd Sunday of March
    #     dst_end = sum(months[:10]) + calendar.monthcalendar(dt.year, 11)[0][6]  # 1st Sunday of November
    #     return dst_start <= day_of_year < dst_end

class SolarCalculator:
    def __init__(self, latitude: float):
        self.latitude = radians(latitude)

    def calculate_angles(self, day_of_year: int, time_of_day: int) -> Tuple[float, float]:
        """Calculate solar azimuth and zenith angles"""
        alpha = 2 * pi * (time_of_day - 43200) / 86400
        delta = radians(23.44 * sin(radians((360/365.25) * (day_of_year-80))))
        
        # Calculate zenith
        chi = acos(sin(delta) * sin(self.latitude) + 
                  cos(delta) * cos(self.latitude) * cos(alpha))
        
        # Calculate azimuth
        tan_xi = sin(alpha) / (sin(self.latitude) * cos(alpha) - 
                              cos(self.latitude) * tan(delta))
        
        # Determine correct quadrant
        xi = self._determine_azimuth_quadrant(alpha, tan_xi)
        
        return degrees(xi), degrees(chi)

    def _determine_azimuth_quadrant(self, alpha: float, tan_xi: float) -> float:
        """Determine correct quadrant for azimuth angle"""
        if alpha > 0:
            return pi + atan(tan_xi) if tan_xi > 0 else 2*pi + atan(tan_xi)
        return atan(tan_xi) if tan_xi > 0 else pi + atan(tan_xi)

class SunPositionCalculator:
    def __init__(self, 
                 latitude: float = 3.039201080912596,
                 longitude: float = 101.79437355320626,
                 time_zone_longitude: float = 120,
                 north_offset: float = -93.964,
                 image_radius: int = None,
                 image_center: Tuple[int, int] = None,
                 sun_mask_radius_ratio: float = 0.065,  # 5% of image radius
                 image_size: Tuple[int, int] = None):
        """Initialize the sun position calculator
        
        Args:
            latitude: Site latitude in degrees
            longitude: Site longitude in degrees
            time_zone_longitude: Longitude of time zone meridian
            north_offset: North offset angle in degrees
            image_radius: Optional radius of the fisheye image in pixels
            image_center: Optional center coordinates of the fisheye image (x, y)
            sun_mask_radius_ratio: Ratio of sun mask radius to image radius
            image_size: Optional tuple of (height, width) of the image
        """
        self.time_converter = TimeConverter(longitude, time_zone_longitude)
        self.solar_calculator = SolarCalculator(latitude)
        self.north_offset = north_offset
        self.image_size = image_size or (1158, 1172)
        
        # Calculate image properties based on size
        height, width = self.image_size
        self.image_center = image_center or (width // 2, height // 2)
        self.image_radius = image_radius or min(width, height) // 2 - 10
        
        # Calculate sun mask radius based on image radius
        self.sun_mask_radius = max(7, int(self.image_radius * sun_mask_radius_ratio))
    
    def calculate_position(self, dt: datetime.datetime) -> SolarPosition:
        """Calculate complete sun position including coordinates and mask"""
        # Get day of year and time of day
        day_of_year, time_of_day = self.time_converter.convert_datetime(dt)
        
        # Calculate solar angles
        azimuth, zenith = self.solar_calculator.calculate_angles(day_of_year, time_of_day)
        
        # Convert to polar coordinates (ρ, θ)
        rho = zenith/90 * self.image_radius
        
        # θ (theta) adjustment for camera orientation
        theta = azimuth - self.north_offset
        
        # Convert polar to Cartesian coordinates
        theta_rad = radians(theta)
        dx = rho * cos(theta_rad)
        dy = -rho * sin(theta_rad)
        
        # Use very small scale factor for more inward position
        sun_x = round(self.image_center[0] + 0.90 * dx)  # Reduced from 0.15 to 0.12
        sun_y = round(self.image_center[1] + 0.90 * dy)  # Reduced from 0.15 to 0.12
        
        # Ensure sun position is within image bounds
        sun_x = max(self.sun_mask_radius, min(sun_x, self.image_size[1] - self.sun_mask_radius))
        sun_y = max(self.sun_mask_radius, min(sun_y, self.image_size[0] - self.sun_mask_radius))
        
        # Create sun mask (circumsolar area)
        mask = self._create_sun_mask(sun_x, sun_y)
        
        # Save angles to CSV for analysis
        self._save_angles_to_csv(dt, azimuth, zenith)
        
        return SolarPosition(sun_x, sun_y, azimuth, zenith, mask)
    
    def _create_sun_mask(self, sun_x: int, sun_y: int) -> np.ndarray:
        """Create binary mask for sun position including circumsolar area
        
        The sun mask is used to:
        1. Avoid misclassification in cloud detection
        2. Define the circumsolar region (proportional to image size)
        3. Help with NRBR+CSL method switching
        """
        height, width = self.image_size
        mask = np.zeros((height, width), dtype=np.uint8)
        y, x = np.ogrid[:height, :width]
        
        # Create circular mask for circumsolar area
        sun_area = (x - sun_x)**2 + (y - sun_y)**2 <= self.sun_mask_radius**2
        
        # Only include sun area within the fisheye circle
        fisheye_area = (x - self.image_center[0])**2 + (y - self.image_center[1])**2 <= self.image_radius**2
        mask[sun_area & fisheye_area] = 255
        
        return mask
    
    def _save_angles_to_csv(self, dt: datetime.datetime, azimuth: float, 
                           zenith: float, filename: str = "sun_angles.csv") -> None:
        """Save solar angles to CSV file"""
        try:
            with open(filename, 'x', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Azimuth (°)", "Zenith (°)"])
        except FileExistsError:
            pass
            
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dt.strftime("%Y-%m-%d %H:%M:%S"), azimuth, zenith])

# Example usage
if __name__ == "__main__":
    calculator = SunPositionCalculator()
    test_dates = [
        datetime.datetime(2025, 2, 11, 11, 31, 0),  # Test with actual image time
    ]
    
    for date in test_dates:
        position = calculator.calculate_position(date)
        print(f"Date: {date}")
        print(f"Sun position (x, y): ({position.x}, {position.y})")
        print(f"Angles (azimuth, zenith): ({position.azimuth:.2f}°, {position.zenith:.2f}°)")
        print("-" * 50)