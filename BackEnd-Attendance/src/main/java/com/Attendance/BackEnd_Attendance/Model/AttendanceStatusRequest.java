package com.Attendance.BackEnd_Attendance.Model;

import java.time.LocalDate;
import java.time.LocalTime;

public class AttendanceStatusRequest {
    private String employeeId;
    private String employeeName;
    private LocalDate date;
    private String status;
    private String reason;
    private LocalTime permissionFrom;
    private LocalTime permissionTo;
    private String halfDayType;
    private LocalTime halfDayFrom;
    private LocalTime halfDayTo;
    private String employeeGender;
    private Boolean markCheckoutTime;
    private LocalTime checkoutTime;
    private Boolean isAdminOverride;
    private Boolean hasExistingStatus;
    private Integer permissionHours;
    private LocalTime fromTime;
    private LocalTime toTime;


    public LocalTime getFromTime() {
        return fromTime;
    }

    public void setFromTime(LocalTime fromTime) {
        this.fromTime = fromTime;
    }

    public Boolean getAdminOverride() {
        return isAdminOverride;
    }

    public void setAdminOverride(Boolean adminOverride) {
        isAdminOverride = adminOverride;
    }

    public Integer getPermissionHours() {
        return permissionHours;
    }

    public void setPermissionHours(Integer permissionHours) {
        this.permissionHours = permissionHours;
    }

    public LocalTime getToTime() {
        return toTime;
    }

    public void setToTime(LocalTime toTime) {
        this.toTime = toTime;
    }

    // Default constructor
    public AttendanceStatusRequest() {}

    // Getters and Setters
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

    public LocalDate getDate() {
        return date;
    }

    public void setDate(LocalDate date) {
        this.date = date;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getReason() {
        return reason;
    }

    public void setReason(String reason) {
        this.reason = reason;
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

    public String getHalfDayType() {
        return halfDayType;
    }

    public void setHalfDayType(String halfDayType) {
        this.halfDayType = halfDayType;
    }

    public LocalTime getHalfDayFrom() {
        return halfDayFrom;
    }

    public void setHalfDayFrom(LocalTime halfDayFrom) {
        this.halfDayFrom = halfDayFrom;
    }

    public LocalTime getHalfDayTo() {
        return halfDayTo;
    }

    public void setHalfDayTo(LocalTime halfDayTo) {
        this.halfDayTo = halfDayTo;
    }

    public String getEmployeeGender() {
        return employeeGender;
    }

    public void setEmployeeGender(String employeeGender) {
        this.employeeGender = employeeGender;
    }

    public Boolean getMarkCheckoutTime() {
        return markCheckoutTime;
    }

    public void setMarkCheckoutTime(Boolean markCheckoutTime) {
        this.markCheckoutTime = markCheckoutTime;
    }

    public LocalTime getCheckoutTime() {
        return checkoutTime;
    }

    public void setCheckoutTime(LocalTime checkoutTime) {
        this.checkoutTime = checkoutTime;
    }

    public Boolean getIsAdminOverride() {
        return isAdminOverride;
    }

    public void setIsAdminOverride(Boolean isAdminOverride) {
        this.isAdminOverride = isAdminOverride;
    }

    public Boolean getHasExistingStatus() {
        return hasExistingStatus;
    }

    public void setHasExistingStatus(Boolean hasExistingStatus) {
        this.hasExistingStatus = hasExistingStatus;
    }


    public AttendanceStatusRequest(String employeeId, LocalDate date, String status, String reason) {
        this.employeeId = employeeId;
        this.date = date;
        this.status = status;
        this.reason = reason;
    }
    @Override
    public String toString() {
        return "AttendanceStatusRequest{" +
                "employeeId='" + employeeId + '\'' +
                ", employeeName='" + employeeName + '\'' +
                ", date=" + date +
                ", status='" + status + '\'' +
                ", reason='" + reason + '\'' +
                ", permissionFrom=" + permissionFrom +
                ", permissionTo=" + permissionTo +
                ", halfDayType='" + halfDayType + '\'' +
                ", halfDayFrom=" + halfDayFrom +
                ", halfDayTo=" + halfDayTo +
                ", employeeGender='" + employeeGender + '\'' +
                ", markCheckoutTime=" + markCheckoutTime +
                ", checkoutTime=" + checkoutTime +
                ", isAdminOverride=" + isAdminOverride +
                ", hasExistingStatus=" + hasExistingStatus +
                '}';
    }




}