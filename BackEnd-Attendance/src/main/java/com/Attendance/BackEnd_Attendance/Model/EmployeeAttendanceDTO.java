package com.Attendance.BackEnd_Attendance.Model;

import java.time.LocalDate;
import java.time.LocalTime;

public class EmployeeAttendanceDTO {
    private String employeeId;
    private String employeeName;
    private String department;
    private LocalDate date;
    private LocalTime checkinTime;
    private LocalTime checkoutTime;
    private String status;

    public EmployeeAttendanceDTO(String employeeId, String employeeName, String department,
                                 LocalDate date, LocalTime checkinTime, LocalTime checkoutTime,
                                 String status) {
        this.employeeId = employeeId;
        this.employeeName = employeeName;
        this.department = department;
        this.date = date;
        this.checkinTime = checkinTime;
        this.checkoutTime = checkoutTime;
        this.status = status;
    }

    public LocalTime getCheckinTime() {
        return checkinTime;
    }

    public void setCheckinTime(LocalTime checkinTime) {
        this.checkinTime = checkinTime;
    }

    public LocalTime getCheckoutTime() {
        return checkoutTime;
    }

    public void setCheckoutTime(LocalTime checkoutTime) {
        this.checkoutTime = checkoutTime;
    }

    public LocalDate getDate() {
        return date;
    }

    public void setDate(LocalDate date) {
        this.date = date;
    }

    public String getDepartment() {
        return department;
    }

    public void setDepartment(String department) {
        this.department = department;
    }

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

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }
// getters and setters
}
