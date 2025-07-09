package com.Attendance.BackEnd_Attendance.Model;

public class AttendanceResponse {
    private String status;
    private String message;
    private String employeeName;
    private String employeeId;
    private String department;
    private String designation;
    private boolean alreadyMarked;
    private String existingStatus;
    private String checkinTime;
    private String checkoutTime;
    private String newCheckinTime;
    private String newStatus;
    private String currentDate;

    // Default constructor
    public AttendanceResponse() {}

    // Constructor for success response
    public AttendanceResponse(String status, String message, String employeeName, String employeeId,
                              String department, String designation, boolean alreadyMarked,
                              String existingStatus, String checkinTime, String checkoutTime,
                              String newCheckinTime, String currentDate) {
        this.status = status;
        this.message = message;
        this.employeeName = employeeName;
        this.employeeId = employeeId;
        this.department = department;
        this.designation = designation;
        this.alreadyMarked = alreadyMarked;
        this.existingStatus = existingStatus;
        this.checkinTime = checkinTime;
        this.checkoutTime = checkoutTime;
        this.newCheckinTime = newCheckinTime;
        this.currentDate = currentDate;
    }

    // Constructor for already marked response
    public AttendanceResponse(String status, String message, String employeeName,
                              boolean alreadyMarked, String existingStatus, String checkinTime) {
        this.status = status;
        this.message = message;
        this.employeeName = employeeName;
        this.alreadyMarked = alreadyMarked;
        this.existingStatus = existingStatus;
        this.checkinTime = checkinTime;
    }

    // Getters and Setters
    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String getEmployeeName() {
        return employeeName;
    }

    public void setEmployeeName(String employeeName) {
        this.employeeName = employeeName;
    }

    public String getEmployeeId() {
        return employeeId;
    }

    public void setEmployeeId(String employeeId) {
        this.employeeId = employeeId;
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

    public boolean isAlreadyMarked() {
        return alreadyMarked;
    }

    public void setAlreadyMarked(boolean alreadyMarked) {
        this.alreadyMarked = alreadyMarked;
    }

    public String getExistingStatus() {
        return existingStatus;
    }

    public void setExistingStatus(String existingStatus) {
        this.existingStatus = existingStatus;
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

    public String getNewCheckinTime() {
        return newCheckinTime;
    }

    public void setNewCheckinTime(String newCheckinTime) {
        this.newCheckinTime = newCheckinTime;
    }

    public String getNewStatus() {
        return newStatus;
    }

    public void setNewStatus(String newStatus) {
        this.newStatus = newStatus;
    }

    public String getCurrentDate() {
        return currentDate;
    }

    public void setCurrentDate(String currentDate) {
        this.currentDate = currentDate;
    }

    @Override
    public String toString() {
        return "AttendanceResponse{" +
                "status='" + status + '\'' +
                ", message='" + message + '\'' +
                ", employeeName='" + employeeName + '\'' +
                ", employeeId='" + employeeId + '\'' +
                ", department='" + department + '\'' +
                ", designation='" + designation + '\'' +
                ", alreadyMarked=" + alreadyMarked +
                ", existingStatus='" + existingStatus + '\'' +
                ", checkinTime='" + checkinTime + '\'' +
                ", checkoutTime='" + checkoutTime + '\'' +
                ", newCheckinTime='" + newCheckinTime + '\'' +
                ", newStatus='" + newStatus + '\'' +
                ", currentDate='" + currentDate + '\'' +
                '}';
    }
}