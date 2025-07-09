package com.Attendance.BackEnd_Attendance.Model;

public class AttendanceStatusResponse {
    private String employeeId;
    private String date;
    private boolean hasCheckedIn;
    private String status;
    private String checkinTime;
    private String reason;

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public String getReason() {
        return reason;
    }

    public void setReason(String reason) {
        this.reason = reason;
    }

    public String getEmployeeId() {
        return employeeId;
    }

    public void setEmployeeId(String employeeId) {
        this.employeeId = employeeId;
    }

    private String checkoutTime;

    // Getters and setters
    public boolean isHasCheckedIn() { return hasCheckedIn; }
    public void setHasCheckedIn(boolean hasCheckedIn) { this.hasCheckedIn = hasCheckedIn; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public String getCheckinTime() { return checkinTime; }
    public void setCheckinTime(String checkinTime) { this.checkinTime = checkinTime; }

    public String getCheckoutTime() { return checkoutTime; }
    public void setCheckoutTime(String checkoutTime) { this.checkoutTime = checkoutTime; }
}
