package com.Attendance.BackEnd_Attendance.Model;

public class FaceRecognitionRequest {

    private String employeeId;
    private String liveImage; // base64 encoded string from frontend

    public String getEmployeeId() {
        return employeeId;
    }

    public void setEmployeeId(String employeeId) {
        this.employeeId = employeeId;
    }

    public String getLiveImage() {
        return liveImage;
    }

    public void setLiveImage(String liveImage) {
        this.liveImage = liveImage;
    }
}
