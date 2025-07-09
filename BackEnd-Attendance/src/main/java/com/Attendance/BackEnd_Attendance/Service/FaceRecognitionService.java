package com.Attendance.BackEnd_Attendance.Service;

import org.springframework.stereotype.Service;

import java.util.Base64;

@Service
public class FaceRecognitionService {

    // Compare the base64-encoded live image with the stored image bytes
    public boolean compareFaces(String liveImageBase64, byte[] storedImage) {
        if (liveImageBase64 == null || storedImage == null) return false;

        try {
            // Clean up the base64 string (remove data URL prefix if present)
            if (liveImageBase64.contains(",")) {
                liveImageBase64 = liveImageBase64.substring(liveImageBase64.indexOf(",") + 1);
            }

            // Decode base64 string to byte array
            byte[] liveImageBytes = Base64.getDecoder().decode(liveImageBase64);

            // Dummy comparison: you can plug in real face recognition logic here
            return java.util.Arrays.equals(liveImageBytes, storedImage);

        } catch (IllegalArgumentException e) {
            System.err.println("Failed to decode base64 image: " + e.getMessage());
            return false;
        }
    }
}
