package com.Attendance.BackEnd_Attendance.Model;

public class AttendanceSummary {
    private String todayStatus;
    private String overallRate;
    private int monthlyPresent;
    private int monthlyAbsent;
    private int monthlyLate;
    private String monthlyAverage;
    private int monthlyPermission;

    // Getters and setters
    public String getTodayStatus() { return todayStatus; }
    public void setTodayStatus(String todayStatus) { this.todayStatus = todayStatus; }

    public String getOverallRate() { return overallRate; }
    public void setOverallRate(String overallRate) { this.overallRate = overallRate; }

    public int getMonthlyPresent() { return monthlyPresent; }
    public void setMonthlyPresent(int monthlyPresent) { this.monthlyPresent = monthlyPresent; }

    public int getMonthlyAbsent() { return monthlyAbsent; }
    public void setMonthlyAbsent(int monthlyAbsent) { this.monthlyAbsent = monthlyAbsent; }

    public int getMonthlyLate() { return monthlyLate; }
    public void setMonthlyLate(int monthlyLate) { this.monthlyLate = monthlyLate; }

    public String getMonthlyAverage() { return monthlyAverage; }
    public void setMonthlyAverage(String monthlyAverage) { this.monthlyAverage = monthlyAverage; }

    public int getMonthlyPermission() { return monthlyPermission; }
    public void setMonthlyPermission(int monthlyPermission) { this.monthlyPermission = monthlyPermission; }
}


