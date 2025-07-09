package com.Attendance.BackEnd_Attendance.Model;

public class PermissionStatusDTO {
    private int usedMinutes;
    private int remainingMinutes;
    private int permissionCount;
    private int remainingPermissions;
    private boolean canRequestPermission;
    private String message;

    public PermissionStatusDTO() {}

    public PermissionStatusDTO(int usedMinutes, int remainingMinutes, int permissionCount,
                               int remainingPermissions, boolean canRequestPermission, String message) {
        this.usedMinutes = usedMinutes;
        this.remainingMinutes = remainingMinutes;
        this.permissionCount = permissionCount;
        this.remainingPermissions = remainingPermissions;
        this.canRequestPermission = canRequestPermission;
        this.message = message;
    }

    // Getters and setters
    public int getUsedMinutes() { return usedMinutes; }
    public void setUsedMinutes(int usedMinutes) { this.usedMinutes = usedMinutes; }

    public int getRemainingMinutes() { return remainingMinutes; }
    public void setRemainingMinutes(int remainingMinutes) { this.remainingMinutes = remainingMinutes; }

    public int getPermissionCount() { return permissionCount; }
    public void setPermissionCount(int permissionCount) { this.permissionCount = permissionCount; }

    public int getRemainingPermissions() { return remainingPermissions; }
    public void setRemainingPermissions(int remainingPermissions) { this.remainingPermissions = remainingPermissions; }

    public boolean isCanRequestPermission() { return canRequestPermission; }
    public void setCanRequestPermission(boolean canRequestPermission) { this.canRequestPermission = canRequestPermission; }

    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }
}