package com.Attendance.BackEnd_Attendance.Model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;

@Entity
public class Employee {

    @Id
    private String id;

    private String usertype;
    private String department;
    private String designation;
    private String username;
    private String gender;
    private String email;
    private String password;

    @Lob
    @Column(name = "profile_image")
    private byte[] profileImage;

    // Default constructor
    public Employee() {
    }

    // Constructor with all fields
    public Employee(String id, String usertype, String department, String designation,
                    String username, String gender, String email, String password, byte[] profileImage) {
        this.id = id;
        this.usertype = usertype;
        this.department = department;
        this.designation = designation;
        this.username = username;
        this.gender = gender;
        this.email = email;
        this.password = password;
        this.profileImage = profileImage;
    }

    // Getters and Setters
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getUsertype() {
        return usertype;
    }

    public void setUsertype(String usertype) {
        this.usertype = usertype;
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

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getGender() {
        return gender;
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public byte[] getProfileImage() {
        return profileImage;
    }

    public void setProfileImage(byte[] profileImage) {
        this.profileImage = profileImage;
    }

    @Override
    public String toString() {
        return "Employee{" +
                "id='" + id + '\'' +
                ", usertype='" + usertype + '\'' +
                ", department='" + department + '\'' +
                ", designation='" + designation + '\'' +
                ", username='" + username + '\'' +
                ", gender='" + gender + '\'' +
                ", email='" + email + '\'' +
                ", password='" + password + '\'' +
                ", profileImage=" + (profileImage != null ? "byte[" + profileImage.length + "]" : "null") +
                '}';
    }
}