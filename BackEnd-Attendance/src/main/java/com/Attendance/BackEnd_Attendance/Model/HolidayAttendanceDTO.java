package com.Attendance.BackEnd_Attendance.Model;


import java.time.LocalDate;

public class HolidayAttendanceDTO {
    private Long employeeId;
    private LocalDate date;
    private String status;
    private String reason;
    private Boolean isHoliday;

    // Constructors
    public HolidayAttendanceDTO() {}

    public HolidayAttendanceDTO(Long employeeId, LocalDate date, String status, String reason, Boolean isHoliday) {
        this.employeeId = employeeId;
        this.date = date;
        this.status = status;
        this.reason = reason;
        this.isHoliday = isHoliday;
    }

    // Getters and setters
    public Long getEmployeeId() { return employeeId; }
    public void setEmployeeId(Long employeeId) { this.employeeId = employeeId; }

    public LocalDate getDate() { return date; }
    public void setDate(LocalDate date) { this.date = date; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public String getReason() { return reason; }
    public void setReason(String reason) { this.reason = reason; }

    public Boolean getIsHoliday() { return isHoliday; }
    public void setIsHoliday(Boolean isHoliday) { this.isHoliday = isHoliday; }
}
