package com.Attendance.BackEnd_Attendance.Model;

public class EmployeePermissionResponse {
    // Basic fields
    private String employeeId;
    private String date;
    private String status;
    private boolean hasCheckedIn;

    // Permission fields
    private boolean hasPermission;
    private String permissionFrom;
    private String permissionTo;
    private String permissionDetails;

    // Half-day fields
    private boolean hasHalfDay;
    private String halfdayType;
    private String halfdayFrom;
    private String halfdayTo;
    private String halfdayDetails;

    // Time fields
    private String checkinTime;
    private String checkoutTime;

    // Default constructor
    public EmployeePermissionResponse() {}

    // Getters and Setters
    public String getEmployeeId() {
        return employeeId;
    }

    public void setEmployeeId(String employeeId) {
        this.employeeId = employeeId;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public boolean isHasCheckedIn() {
        return hasCheckedIn;
    }

    public void setHasCheckedIn(boolean hasCheckedIn) {
        this.hasCheckedIn = hasCheckedIn;
    }

    public boolean isHasPermission() {
        return hasPermission;
    }

    public void setHasPermission(boolean hasPermission) {
        this.hasPermission = hasPermission;
    }

    public String getPermissionFrom() {
        return permissionFrom;
    }

    public void setPermissionFrom(String permissionFrom) {
        this.permissionFrom = permissionFrom;
    }

    public String getPermissionTo() {
        return permissionTo;
    }

    public void setPermissionTo(String permissionTo) {
        this.permissionTo = permissionTo;
    }

    public String getPermissionDetails() {
        return permissionDetails;
    }

    public void setPermissionDetails(String permissionDetails) {
        this.permissionDetails = permissionDetails;
    }

    public boolean isHasHalfDay() {
        return hasHalfDay;
    }

    public void setHasHalfDay(boolean hasHalfDay) {
        this.hasHalfDay = hasHalfDay;
    }

    public String getHalfdayType() {
        return halfdayType;
    }

    public void setHalfdayType(String halfdayType) {
        this.halfdayType = halfdayType;
    }

    public String getHalfdayFrom() {
        return halfdayFrom;
    }

    public void setHalfdayFrom(String halfdayFrom) {
        this.halfdayFrom = halfdayFrom;
    }

    public String getHalfdayTo() {
        return halfdayTo;
    }

    public void setHalfdayTo(String halfdayTo) {
        this.halfdayTo = halfdayTo;
    }

    public String getHalfdayDetails() {
        return halfdayDetails;
    }

    public void setHalfdayDetails(String halfdayDetails) {
        this.halfdayDetails = halfdayDetails;
    }

    public String getCheckinTime() {
        return checkinTime;
    }

    public void setCheckinTime(String checkinTime) {
        this.checkinTime = checkinTime;
    }

    public String getCheckoutTime() {
        return checkoutTime;
    }

    public void setCheckoutTime(String checkoutTime) {
        this.checkoutTime = checkoutTime;
    }

    @Override
    public String toString() {
        return "EmployeePermissionResponse{" +
                "employeeId='" + employeeId + '\'' +
                ", date='" + date + '\'' +
                ", status='" + status + '\'' +
                ", hasCheckedIn=" + hasCheckedIn +
                ", hasPermission=" + hasPermission +
                ", permissionFrom='" + permissionFrom + '\'' +
                ", permissionTo='" + permissionTo + '\'' +
                ", hasHalfDay=" + hasHalfDay +
                ", halfdayType='" + halfdayType + '\'' +
                ", halfdayFrom='" + halfdayFrom + '\'' +
                ", halfdayTo='" + halfdayTo + '\'' +
                ", checkinTime='" + checkinTime + '\'' +
                ", checkoutTime='" + checkoutTime + '\'' +
                '}';
    }
}