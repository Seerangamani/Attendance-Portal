package com.Attendance.BackEnd_Attendance.Model;

import jakarta.persistence.*;
import java.time.LocalDate;
import java.time.LocalTime;

@Entity
@Table(name = "attendance")
public class Attendance {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "employee_id")
    private String employeeId;

    @Column(name = "employee_name")
    private String employeeName;

    @Column(name = "department")
    private String department;

    @Column(name = "designation")
    private String designation;

    @Column(name = "date")
    private LocalDate date;

    @Column(name = "checkin_time")
    private LocalTime checkinTime;

    @Column(name = "checkout_time")
    private LocalTime checkoutTime;

    @Column(name = "status")
    private String status;

    @Column(name = "permission_from")
    private LocalTime permissionFrom;

    @Column(name = "permission_to")
    private LocalTime permissionTo;

    @Column(name = "halfday_from")
    private LocalTime halfdayFrom;

    @Column(name = "halfday_to")
    private LocalTime halfdayTo;  // Fixed the typo from 'haldayTo'

    @Column(name = "halfday_type")
    private String halfdayType;

    @Column(name = "reason")
    private String reason;

    // Default constructor
    public Attendance() {}

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
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

    public LocalDate getDate() {
        return date;
    }

    public void setDate(LocalDate date) {
        this.date = date;
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

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public LocalTime getPermissionFrom() {
        return permissionFrom;
    }

    public void setPermissionFrom(LocalTime permissionFrom) {
        this.permissionFrom = permissionFrom;
    }

    public LocalTime getPermissionTo() {
        return permissionTo;
    }

    public void setPermissionTo(LocalTime permissionTo) {
        this.permissionTo = permissionTo;
    }

    public LocalTime getHalfdayFrom() {
        return halfdayFrom;
    }

    public void setHalfdayFrom(LocalTime halfdayFrom) {
        this.halfdayFrom = halfdayFrom;
    }

    public LocalTime getHalfdayTo() {
        return halfdayTo;
    }

    public void setHalfdayTo(LocalTime halfdayTo) {
        this.halfdayTo = halfdayTo;
    }

    public String getHalfdayType() {
        return halfdayType;
    }

    public void setHalfdayType(String halfdayType) {
        this.halfdayType = halfdayType;
    }

    public String getReason() {
        return reason;
    }

    public void setReason(String reason) {
        this.reason = reason;
    }

    @Override
    public String toString() {
        return "Attendance{" +
                "id=" + id +
                ", employeeId='" + employeeId + '\'' +
                ", employeeName='" + employeeName + '\'' +
                ", department='" + department + '\'' +
                ", designation='" + designation + '\'' +
                ", date=" + date +
                ", checkinTime=" + checkinTime +
                ", checkoutTime=" + checkoutTime +
                ", status='" + status + '\'' +
                ", permissionFrom=" + permissionFrom +
                ", permissionTo=" + permissionTo +
                ", halfdayFrom=" + halfdayFrom +
                ", halfdayTo=" + halfdayTo +
                ", halfdayType='" + halfdayType + '\'' +
                ", reason='" + reason + '\'' +
                '}';
    }
}