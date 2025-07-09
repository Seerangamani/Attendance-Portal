package com.Attendance.BackEnd_Attendance.Model;

import java.time.LocalDate;

public class HolidayRequest {
    private LocalDate date;
    private String reason;
    private String status;
    private Boolean isHoliday;
    private Boolean isAdminOverride;

    // Getters and setters
    public LocalDate getDate() { return date; }
    public void setDate(LocalDate date) { this.date = date; }

    public String getReason() { return reason; }
    public void setReason(String reason) { this.reason = reason; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public Boolean getIsHoliday() { return isHoliday; }
    public void setIsHoliday(Boolean isHoliday) { this.isHoliday = isHoliday; }

    public Boolean getIsAdminOverride() { return isAdminOverride; }
    public void setIsAdminOverride(Boolean isAdminOverride) { this.isAdminOverride = isAdminOverride; }
}