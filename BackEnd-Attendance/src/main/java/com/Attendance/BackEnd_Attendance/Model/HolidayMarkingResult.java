package com.Attendance.BackEnd_Attendance.Model;

public class HolidayMarkingResult {
    private String message;
    private int employeesMarked;
    private int existingRecordsUpdated;

    // Default constructor
    public HolidayMarkingResult() {}

    // Constructor with parameters
    public HolidayMarkingResult(String message, int employeesMarked, int existingRecordsUpdated) {
        this.message = message;
        this.employeesMarked = employeesMarked;
        this.existingRecordsUpdated = existingRecordsUpdated;
    }

    // Getters and setters
    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public int getEmployeesMarked() {
        return employeesMarked;
    }

    public void setEmployeesMarked(int employeesMarked) {
        this.employeesMarked = employeesMarked;
    }

    public int getExistingRecordsUpdated() {
        return existingRecordsUpdated;
    }

    public void setExistingRecordsUpdated(int existingRecordsUpdated) {
        this.existingRecordsUpdated = existingRecordsUpdated;
    }

    @Override
    public String toString() {
        return "HolidayMarkingResult{" +
                "message='" + message + '\'' +
                ", employeesMarked=" + employeesMarked +
                ", existingRecordsUpdated=" + existingRecordsUpdated +
                '}';
    }
}