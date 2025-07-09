package com.Attendance.BackEnd_Attendance.Model;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;

public class AttendanceRequest {
    private String employeeId;
    private String employeeName;
    private String department;
    private String designation;
    private String gender;
    private String status;
    private String halfdayType;
    private String time;
    private LocalTime permissionFrom;
    private LocalTime permissionTo;
    private String action;
    private LocalTime halfdayFrom;
    private LocalTime halfdayTo;
    private boolean isLateArrival;
    private boolean hasHalfDay;
    private boolean hasPermission;

    // New methods for attendance logic
    public boolean isHasHalfDay() {
        return hasHalfDay;
    }

    public void setHasHalfDay(boolean hasHalfDay) {
        this.hasHalfDay = hasHalfDay;
    }

    public boolean isHasPermission() {
        return hasPermission;
    }

    public void setHasPermission(boolean hasPermission) {
        this.hasPermission = hasPermission;
    }

    // Improved time conversion with error handling
    public LocalTime getTimeAsLocalTime() {
        if (time == null || time.isEmpty()) {
            return null;
        }

        try {
            // Handle different time formats
            if (time.length() >= 19) {
                // Assuming format: "yyyy-MM-dd HH:mm:ss"
                return LocalTime.parse(time.substring(11, 19));
            } else if (time.length() >= 8) {
                // Assuming format: "HH:mm:ss"
                return LocalTime.parse(time);
            } else if (time.length() >= 5) {
                // Assuming format: "HH:mm"
                return LocalTime.parse(time, DateTimeFormatter.ofPattern("HH:mm"));
            }
        } catch (DateTimeParseException e) {
            // Log error or handle gracefully
            System.err.println("Failed to parse time: " + time);
        }

        return null;
    }

    // Validation methods
    public boolean isValidPermissionTime() {
        return permissionFrom != null && permissionTo != null &&
                permissionFrom.isBefore(permissionTo);
    }

    public boolean isValidHalfdayTime() {
        return halfdayFrom != null && halfdayTo != null &&
                halfdayFrom.isBefore(halfdayTo);
    }

    // Existing getters and setters
    public String getEmployeeId() {
        return employeeId;
    }

    public void setEmployeeId(String employeeId) {
        this.employeeId = employeeId;
    }

    public String getEmployeeName() {
        return employeeName;
    }

    public void setEmployeeName(String employeeName) {
        this.employeeName = employeeName;
    }

    public String getDepartment() {
        return department;
    }

    public void setDepartment(String department) {
        this.department = department;
    }

    public String getDesignation() {
        return designation;
    }

    public void setDesignation(String designation) {
        this.designation = designation;
    }

    public String getGender() {
        return gender;
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getHalfdayType() {
        return halfdayType;
    }

    public void setHalfdayType(String halfdayType) {
        this.halfdayType = halfdayType;
    }

    public String getTime() {
        return time;
    }

    public void setTime(String time) {
        this.time = time;
    }

    public LocalTime getPermissionFrom() {
        return permissionFrom;
    }

    public void setPermissionFrom(LocalTime permissionFrom) {
        this.permissionFrom = permissionFrom;
    }

    public LocalTime getPermissionTo() {
        return permissionTo;
    }

    public void setPermissionTo(LocalTime permissionTo) {
        this.permissionTo = permissionTo;
    }

    public String getAction() {
        return action;
    }

    public void setAction(String action) {
        this.action = action;
    }

    public LocalTime getHalfdayFrom() {
        return halfdayFrom;
    }

    public void setHalfdayFrom(LocalTime halfdayFrom) {
        this.halfdayFrom = halfdayFrom;
    }

    public LocalTime getHalfdayTo() {
        return halfdayTo;
    }

    public void setHalfdayTo(LocalTime halfdayTo) {
        this.halfdayTo = halfdayTo;
    }

    public boolean isLateArrival() {
        return isLateArrival;
    }

    public void setLateArrival(boolean lateArrival) {
        isLateArrival = lateArrival;
    }

    @Override
    public String toString() {
        return "AttendanceRequest{" +
                "employeeId='" + employeeId + '\'' +
                ", employeeName='" + employeeName + '\'' +
                ", department='" + department + '\'' +
                ", hasHalfDay=" + hasHalfDay +
                ", hasPermission=" + hasPermission +
                ", status='" + status + '\'' +
                '}';
    }
}