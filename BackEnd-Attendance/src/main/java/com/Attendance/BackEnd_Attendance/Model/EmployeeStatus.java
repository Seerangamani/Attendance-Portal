// BACKEND - EmployeeStatus DTO
package com.Attendance.BackEnd_Attendance.Model;

public class EmployeeStatus {
    private String employeeId;
    private String employeeName;
    private String department;
    private String designation;
    private String status; // Present, Late, Absent

    public EmployeeStatus() {}

    public EmployeeStatus(String employeeId, String employeeName, String department, String designation, String status) {
        this.employeeId = employeeId;
        this.employeeName = employeeName;
        this.department = department;
        this.designation = designation;
        this.status = status;
    }

    // Getters and Setters
    public String getEmployeeId() { return employeeId; }
    public void setEmployeeId(String employeeId) { this.employeeId = employeeId; }

    public String getEmployeeName() { return employeeName; }
    public void setEmployeeName(String employeeName) { this.employeeName = employeeName; }

    public String getDepartment() { return department; }
    public void setDepartment(String department) { this.department = department; }

    public String getDesignation() { return designation; }
    public void setDesignation(String designation) { this.designation = designation; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
}
