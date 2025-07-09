package com.Attendance.BackEnd_Attendance.Model;

public class EmployeeStatusResponse {
    private String employeeId;
    private boolean hasPermission;
    private boolean hasHalfDay;
    private boolean hasCheckedIn;
    private String currentStatus;
    private String permissionDetails;
    private String halfdayDetails;

    // Getters and setters
    public String getEmployeeId() { return employeeId; }
    public void setEmployeeId(String employeeId) { this.employeeId = employeeId; }

    public boolean isHasPermission() { return hasPermission; }
    public void setHasPermission(boolean hasPermission) { this.hasPermission = hasPermission; }

    public boolean isHasHalfDay() { return hasHalfDay; }
    public void setHasHalfDay(boolean hasHalfDay) { this.hasHalfDay = hasHalfDay; }

    public boolean isHasCheckedIn() { return hasCheckedIn; }
    public void setHasCheckedIn(boolean hasCheckedIn) { this.hasCheckedIn = hasCheckedIn; }

    public String getCurrentStatus() { return currentStatus; }
    public void setCurrentStatus(String currentStatus) { this.currentStatus = currentStatus; }

    public String getPermissionDetails() { return permissionDetails; }
    public void setPermissionDetails(String permissionDetails) { this.permissionDetails = permissionDetails; }

    public String getHalfdayDetails() { return halfdayDetails; }
    public void setHalfdayDetails(String halfdayDetails) { this.halfdayDetails = halfdayDetails; }

    public void setCheckinTime(String string) {
    }
}
