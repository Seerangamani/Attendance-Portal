//package com.Attendance.BackEnd_Attendance.Model;
//
//import com.fasterxml.jackson.annotation.JsonProperty;
//
//public class AttendanceStatsDTO {
//
//    @JsonProperty("employeeId")
//    private String employeeId;
//
//    @JsonProperty("present")
//    private int present;
//
//    @JsonProperty("absent")
//    private int absent;
//
//    @JsonProperty("late")
//    private int late;
//
//    @JsonProperty("onduty")
//    private int onduty;
//
//    @JsonProperty("halfday")
//    private int halfday;
//
//    @JsonProperty("permission")
//    private int permission;
//
//    @JsonProperty("total")
//    private int total;
//
//    // Constructors
//    public AttendanceStatsDTO() {}
//
//    public AttendanceStatsDTO(String employeeId, int present, int absent, int late,
//                              int onduty, int halfday, int permission, int total) {
//        this.employeeId = employeeId;
//        this.present = present;
//        this.absent = absent;
//        this.late = late;
//        this.onduty = onduty;
//        this.halfday = halfday;
//        this.permission = permission;
//        this.total = total;
//    }
//
//    // Getters and Setters
//    public String getEmployeeId() {
//        return employeeId;
//    }
//
//    public void setEmployeeId(String employeeId) {
//        this.employeeId = employeeId;
//    }
//
//    public int getPresent() {
//        return present;
//    }
//
//    public void setPresent(int present) {
//        this.present = present;
//    }
//
//    public int getAbsent() {
//        return absent;
//    }
//
//    public void setAbsent(int absent) {
//        this.absent = absent;
//    }
//
//    public int getLate() {
//        return late;
//    }
//
//    public void setLate(int late) {
//        this.late = late;
//    }
//
//    public int getOnduty() {
//        return onduty;
//    }
//
//    public void setOnduty(int onduty) {
//        this.onduty = onduty;
//    }
//
//    public int getHalfday() {
//        return halfday;
//    }
//
//    public void setHalfday(int halfday) {
//        this.halfday = halfday;
//    }
//
//    public int getPermission() {
//        return permission;
//    }
//
//    public void setPermission(int permission) {
//        this.permission = permission;
//    }
//
//    public int getTotal() {
//        return total;
//    }
//
//    public void setTotal(int total) {
//        this.total = total;
//    }
//
//    // Helper methods
//    public double getAttendancePercentage() {
//        if (total == 0) return 0.0;
//        return ((double)(present + late + onduty + halfday + permission) / total) * 100;
//    }
//
//    public int getTotalWorkingDays() {
//        return present + late + onduty + halfday + permission;
//    }
//
//    @Override
//    public String toString() {
//        return "AttendanceStatsDTO{" +
//                "employeeId='" + employeeId + '\'' +
//                ", present=" + present +
//                ", absent=" + absent +
//                ", late=" + late +
//                ", onduty=" + onduty +
//                ", halfday=" + halfday +
//                ", permission=" + permission +
//                ", total=" + total +
//                ", attendancePercentage=" + getAttendancePercentage() +
//                '}';
//    }
//}