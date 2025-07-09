package com.Attendance.BackEnd_Attendance.Model;

public class AdminDashboardStats {
    private long totalEmployees;
    private long presentToday;
    private long onDuty; // Changed from onLeave to onDuty
    private long absent;
    private long lateArrivals;
    private String attendanceRate;
    private long permissionCount;

    // Default constructor
    public AdminDashboardStats() {}

    // Constructor with all fields
    public AdminDashboardStats(long totalEmployees, long presentToday, long onDuty,
                               long absent, long lateArrivals, String attendanceRate, long permissionCount) {
        this.totalEmployees = totalEmployees;
        this.presentToday = presentToday;
        this.onDuty = onDuty;
        this.absent = absent;
        this.lateArrivals = lateArrivals;
        this.attendanceRate = attendanceRate;
        this.permissionCount = permissionCount;
    }

    // Getters and Setters
    public long getTotalEmployees() {
        return totalEmployees;
    }

    public void setTotalEmployees(long totalEmployees) {
        this.totalEmployees = totalEmployees;
    }

    public long getPresentToday() {
        return presentToday;
    }

    public void setPresentToday(long presentToday) {
        this.presentToday = presentToday;
    }

    public long getOnDuty() { // Changed from getOnLeave to getOnDuty
        return onDuty;
    }

    public void setOnDuty(long onDuty) { // Changed from setOnLeave to setOnDuty
        this.onDuty = onDuty;
    }

    public long getAbsent() {
        return absent;
    }

    public void setAbsent(long absent) {
        this.absent = absent;
    }

    public long getLateArrivals() {
        return lateArrivals;
    }

    public void setLateArrivals(long lateArrivals) {
        this.lateArrivals = lateArrivals;
    }

    public String getAttendanceRate() {
        return attendanceRate;
    }

    public void setAttendanceRate(String attendanceRate) {
        this.attendanceRate = attendanceRate;
    }

    public long getPermissionCount() {
        return permissionCount;
    }

    public void setPermissionCount(long permissionCount) {
        this.permissionCount = permissionCount;
    }

    @Override
    public String toString() {
        return "AdminDashboardStats{" +
                "totalEmployees=" + totalEmployees +
                ", presentToday=" + presentToday +
                ", onDuty=" + onDuty +
                ", absent=" + absent +
                ", lateArrivals=" + lateArrivals +
                ", attendanceRate='" + attendanceRate + '\'' +
                ", permissionCount=" + permissionCount +
                '}';
    }
}