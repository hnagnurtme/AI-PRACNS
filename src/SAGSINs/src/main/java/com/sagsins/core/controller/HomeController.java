package com.sagsins.core.controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class HomeController {


    @GetMapping("/")
    public String home(Model model) {
        return "Home";
    }

    @GetMapping("/setPosition")
    @ResponseBody
    public String setPosition(@RequestParam double lat, @RequestParam double lng, @RequestParam double alt) {
        // Log or process the received coordinates as needed
        System.out.println("Received position: lat=" + lat + ", lng=" + lng + ", alt=" + alt);
        return "Position set: lat=" + lat + ", lng=" + lng + ", alt=" + alt;
    }
}
