package com.Attendance.BackEnd_Attendance.Model;


import jakarta.persistence.Entity;
import jakarta.persistence.Id;

@Entity
public class Admin {
    @Id
    private String id;
    private String email;
    private String password;

    public Admin() {}

    public Admin(String id, String email, String password) {
        this.id = id;
        this.email = email;
        this.password = password;
    }

    // Fix getter/setter names to match field names
    public String getId() {  // Changed from getAdminId()
        return id;
    }

    public void setId(String id) {  // Changed from setAdminId()
        this.id = id;
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
}
