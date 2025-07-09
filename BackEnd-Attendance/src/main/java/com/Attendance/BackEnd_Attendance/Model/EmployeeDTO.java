// EmployeeDTO.java
package com.Attendance.BackEnd_Attendance.Model;

public class EmployeeDTO {
    private String id;
    private String username;
    private String email;
    private String department;
    private String designation;
    private String profileImage;

    public EmployeeDTO(String id, String username, String email, String department, String designation, String gender, String employee, String s) {
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

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getProfileImage() {
        return profileImage;
    }

    public void setProfileImage(String profileImage) {
        this.profileImage = profileImage;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public EmployeeDTO(String department, String designation, String email, String id, String profileImage, String username) {
        this.department = department;
        this.designation = designation;
        this.email = email;
        this.id = id;
        this.profileImage = profileImage;
        this.username = username;
    }

}