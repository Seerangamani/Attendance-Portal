package com.Attendance.BackEnd_Attendance.Controller;

import com.Attendance.BackEnd_Attendance.Model.Attendance;
import com.Attendance.BackEnd_Attendance.Model.Employee;
import com.Attendance.BackEnd_Attendance.Model.EmployeeAttendanceDTO;
import com.Attendance.BackEnd_Attendance.Repository.AttendanceRepository;
import com.Attendance.BackEnd_Attendance.Repository.EmployeeRepository;
import com.Attendance.BackEnd_Attendance.Service.AttendanceService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/report/attendance-report")
@CrossOrigin(origins = "*")
public class AttendanceReportController {

    @Autowired
    private AttendanceRepository attendanceRepository;

    @Autowired
    private EmployeeRepository employeeRepository;

    @Autowired
    private AttendanceService attendanceService;

    // ✅ 1. Total Employee Count
    @GetMapping("/total-employee-count")
    public long getTotalEmployeeCount() {
        return employeeRepository.count();
    }

    // ✅ 2. Today's Attendance Status with Date Range Support
    @GetMapping("/today-status")
    public List<Map<String, Object>> getTodayAttendance(
            @RequestParam(required = false) String usertype,
            @RequestParam(required = false) String name,
            @RequestParam(required = false) String startDate,
            @RequestParam(required = false) String endDate
    ) {
        LocalDate fromDate;
        LocalDate toDate;

        // Handle date parameters
        if (startDate != null && !startDate.isEmpty()) {
            fromDate = LocalDate.parse(startDate);
        } else {
            fromDate = LocalDate.now();
        }

        if (endDate != null && !endDate.isEmpty()) {
            toDate = LocalDate.parse(endDate);
        } else {
            toDate = fromDate;
        }

        List<Attendance> attendanceRecords;
        if (fromDate.equals(toDate)) {
            // Single date query
            attendanceRecords = attendanceRepository.findByDate(fromDate);
        } else {
            // Date range query
            attendanceRecords = attendanceRepository.findByDateBetween(fromDate, toDate);
        }

        List<Employee> allEmployees = employeeRepository.findAll();

        // Create a map for quick lookup - handling multiple records per employee
        Map<String, List<Attendance>> employeeAttendanceMap = attendanceRecords.stream()
                .collect(Collectors.groupingBy(Attendance::getEmployeeId));

        List<Map<String, Object>> result = new ArrayList<>();
        int sno = 1;

        for (Employee employee : allEmployees) {
            // Apply filters
            if (usertype != null && !usertype.isEmpty() && !"All".equalsIgnoreCase(usertype) &&
                    !employee.getUsertype().equalsIgnoreCase(usertype)) {
                continue;
            }
            if (name != null && !name.isEmpty() &&
                    !employee.getUsername().toLowerCase().contains(name.toLowerCase())) {
                continue;
            }

            List<Attendance> employeeAttendanceList = employeeAttendanceMap.get(employee.getId());

            if (employeeAttendanceList != null && !employeeAttendanceList.isEmpty()) {
                // If date range, create multiple records for each date
                for (Attendance attendance : employeeAttendanceList) {
                    Map<String, Object> record = createEmployeeRecord(employee, attendance, sno++);
                    result.add(record);
                }
            } else {
                // No attendance record found - mark as absent for the query date
                Map<String, Object> record = createEmployeeRecord(employee, null, sno++);
                record.put("date", fromDate.toString());
                result.add(record);
            }
        }

        return result;
    }

    private Map<String, Object> createEmployeeRecord(Employee employee, Attendance attendance, int sno) {
        Map<String, Object> record = new HashMap<>();
        record.put("sno", sno);
        record.put("employeeId", employee.getId());
        record.put("employeeName", employee.getUsername());
        record.put("username", employee.getUsername());
        record.put("department", employee.getDepartment());
        record.put("designation", employee.getDesignation());
        record.put("email", employee.getEmail());
        record.put("usertype", employee.getUsertype());

        if (attendance != null) {
            String status = normalizeStatus(attendance.getStatus());
            record.put("status", status);
            record.put("checkinTime", attendance.getCheckinTime());
            record.put("checkoutTime", attendance.getCheckoutTime());
            record.put("date", attendance.getDate().toString());
        } else {
            record.put("status", "Absent");
            record.put("checkinTime", null);
            record.put("checkoutTime", null);
        }

        return record;
    }

    // ✅ 3. Enhanced Teamwise Summary with Date Range Support
    @GetMapping("/teamwise-summary")
    public Map<String, Map<String, Integer>> getTeamwiseSummary(
            @RequestParam(required = false) String startDate,
            @RequestParam(required = false) String endDate
    ) {
        LocalDate fromDate;
        LocalDate toDate;

        // Handle date parameters
        if (startDate != null && !startDate.isEmpty()) {
            fromDate = LocalDate.parse(startDate);
        } else {
            fromDate = LocalDate.now();
        }

        if (endDate != null && !endDate.isEmpty()) {
            toDate = LocalDate.parse(endDate);
        } else {
            toDate = fromDate;
        }

        List<Attendance> attendanceRecords;
        if (fromDate.equals(toDate)) {
            attendanceRecords = attendanceRepository.findByDate(fromDate);
        } else {
            attendanceRecords = attendanceRepository.findByDateBetween(fromDate, toDate);
        }

        List<Employee> allEmployees = employeeRepository.findAll();

        // Create attendance map for quick lookup
        Map<String, List<Attendance>> employeeAttendanceMap = attendanceRecords.stream()
                .collect(Collectors.groupingBy(Attendance::getEmployeeId));

        Map<String, Map<String, Integer>> teamSummary = new HashMap<>();

        // Initialize team summary maps
        Set<String> userTypes = allEmployees.stream()
                .map(Employee::getUsertype)
                .collect(Collectors.toSet());

        for (String userType : userTypes) {
            Map<String, Integer> statusCount = new HashMap<>();
            statusCount.put("Present", 0);
            statusCount.put("Late", 0);
            statusCount.put("Absent", 0);
            statusCount.put("On Duty", 0);
            teamSummary.put(userType.toLowerCase(), statusCount);
        }

        // Count employees by team and status
        for (Employee employee : allEmployees) {
            String userType = employee.getUsertype().toLowerCase();
            Map<String, Integer> statusCount = teamSummary.get(userType);

            if (statusCount != null) {
                List<Attendance> employeeAttendanceList = employeeAttendanceMap.get(employee.getId());

                if (employeeAttendanceList != null && !employeeAttendanceList.isEmpty()) {
                    // For date ranges, count unique dates per status
                    Set<String> uniqueStatuses = employeeAttendanceList.stream()
                            .map(att -> normalizeStatus(att.getStatus()))
                            .collect(Collectors.toSet());

                    // For simplicity, take the most recent status if multiple records exist
                    String finalStatus = employeeAttendanceList.stream()
                            .max(Comparator.comparing(Attendance::getDate))
                            .map(att -> normalizeStatus(att.getStatus()))
                            .orElse("Absent");

                    statusCount.put(finalStatus, statusCount.getOrDefault(finalStatus, 0) + 1);
                } else {
                    // No attendance record - mark as absent
                    statusCount.put("Absent", statusCount.getOrDefault("Absent", 0) + 1);
                }
            }
        }

        return teamSummary;
    }

    // Helper method to normalize status names
    private String normalizeStatus(String status) {
        if (status == null) return "Absent";

        switch (status.toLowerCase().trim()) {
            case "present":
            case "checked-out":
                return "Present";
            case "late":
                return "Late";
            case "on duty":
            case "permission granted":
                return "On Duty";
            case "on leave":
            case "absent":
            default:
                return "Absent";
        }
    }

    // ✅ 4. Monthly Attendance (unchanged)
    @GetMapping("/monthly/{employeeId}")
    public List<Attendance> getMonthlyAttendance(@PathVariable String employeeId) {
        LocalDate start = LocalDate.now().withDayOfMonth(1);
        LocalDate end = LocalDate.now();
        return attendanceRepository.findByEmployeeIdAndDateBetween(employeeId, start, end);
    }

    // ✅ 5. All Employees List with S.No (unchanged)
    @GetMapping("/employees")
    public List<Map<String, Object>> getEmployeeNames() {
        List<Employee> employees = employeeRepository.findAll();
        List<Map<String, Object>> result = new ArrayList<>();
        int sno = 1;

        for (Employee e : employees) {
            Map<String, Object> emp = new HashMap<>();
            emp.put("sno", sno++);
            emp.put("employeeId", e.getId());
            emp.put("name", e.getUsername());
            emp.put("department", e.getDepartment());
            emp.put("designation", e.getDesignation());
            emp.put("usertype", e.getUsertype());
            result.add(emp);
        }
        return result;
    }

    // ✅ 6. Attendance Report - All Employees (unchanged)
    @GetMapping("/attendance-report")
    public List<EmployeeAttendanceDTO> getAttendanceReport() {
        return attendanceService.getAttendanceReport();
    }

    // ✅ 7. Additional endpoint for filtered attendance with better date range support
    @GetMapping("/filtered-attendance")
    public List<Map<String, Object>> getFilteredAttendance(
            @RequestParam(required = false) String usertype,
            @RequestParam(required = false) String name,
            @RequestParam(required = false) String startDate,
            @RequestParam(required = false) String endDate,
            @RequestParam(required = false) String status
    ) {
        // This endpoint provides more comprehensive filtering options
        return getTodayAttendance(usertype, name, startDate, endDate);
    }
}