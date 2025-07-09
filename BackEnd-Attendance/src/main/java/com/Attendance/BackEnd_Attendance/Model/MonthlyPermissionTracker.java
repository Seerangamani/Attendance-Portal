package com.Attendance.BackEnd_Attendance.Model;

import jakarta.persistence.*;

import java.time.YearMonth;
import java.util.concurrent.TimeUnit;

// Add this new entity class or update existing MonthlyPermissionTracker
@Entity
@Table(name = "monthly_permission_tracker")
public class MonthlyPermissionTracker {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "employee_id")
    private String employeeId;

    @Column(name = "year_month")
    private String yearMonth; // Format: "2025-06"

    @Column(name = "total_permission_minutes_used")
    private int totalPermissionMinutesUsed;

    @Column(name = "permission_count")
    private int permissionCount;

    @Column(name = "grace_minutes_used")
    private int graceMinutesUsed;

    @Column(name = "first_permission_grace_used")
    private boolean firstPermissionGraceUsed;

    @Column(name = "second_permission_grace_used")
    private boolean secondPermissionGraceUsed;

    // Constructors, getters, and setters
    public MonthlyPermissionTracker() {}

    public MonthlyPermissionTracker(String employeeId, String yearMonth) {
        this.employeeId = employeeId;
        this.yearMonth = yearMonth;
        this.totalPermissionMinutesUsed = 0;
        this.permissionCount = 0;
        this.graceMinutesUsed = 0;
        this.firstPermissionGraceUsed = false;
        this.secondPermissionGraceUsed = false;
    }

    // Getters and setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getEmployeeId() { return employeeId; }
    public void setEmployeeId(String employeeId) { this.employeeId = employeeId; }

    public String getYearMonth() { return yearMonth; }
    public void setYearMonth(String yearMonth) { this.yearMonth = yearMonth; }

    public int getTotalPermissionMinutesUsed() { return totalPermissionMinutesUsed; }
    public void setTotalPermissionMinutesUsed(int totalPermissionMinutesUsed) { this.totalPermissionMinutesUsed = totalPermissionMinutesUsed; }

    public int getPermissionCount() { return permissionCount; }
    public void setPermissionCount(int permissionCount) { this.permissionCount = permissionCount; }

    public int getGraceMinutesUsed() { return graceMinutesUsed; }
    public void setGraceMinutesUsed(int graceMinutesUsed) { this.graceMinutesUsed = graceMinutesUsed; }

    public boolean isFirstPermissionGraceUsed() { return firstPermissionGraceUsed; }
    public void setFirstPermissionGraceUsed(boolean firstPermissionGraceUsed) { this.firstPermissionGraceUsed = firstPermissionGraceUsed; }

    public boolean isSecondPermissionGraceUsed() { return secondPermissionGraceUsed; }
    public void setSecondPermissionGraceUsed(boolean secondPermissionGraceUsed) { this.secondPermissionGraceUsed = secondPermissionGraceUsed; }
}