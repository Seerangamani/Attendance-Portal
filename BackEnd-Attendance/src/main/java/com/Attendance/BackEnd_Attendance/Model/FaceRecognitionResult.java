package com.Attendance.BackEnd_Attendance.Model;

public class FaceRecognitionResult {
    private String message;
    private String status;
    private boolean success; // Add this field

    // Default constructor
    public FaceRecognitionResult() {
    }

    // Constructor with success, message, and status
    public FaceRecognitionResult(boolean success, String message, String status) {
        this.success = success;
        this.message = message;
        this.status = status;
    }

    // Constructor with message and status (success defaults to false)
    public FaceRecognitionResult(String message, String status) {
        this.success = false;
        this.message = message;
        this.status = status;
    }

    // Constructor with message only (status defaults to "error", success to false)
    public FaceRecognitionResult(String message) {
        this.success = false;
        this.message = message;
        this.status = "error";
    }

    // Getters and Setters
    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public boolean isSuccess() {
        return success;
    }

    public void setSuccess(boolean success) {
        this.success = success;
    }
}