package com.sagsins.core.controller;

import java.util.List;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.sagsins.core.model.NodeInfo;
import com.sagsins.core.service.INodeService;

@Controller
public class HomeController {
    private final INodeService nodeService;
    public HomeController(INodeService nodeService) {
        this.nodeService = nodeService;
    }

    @GetMapping("/")
    public String home(Model model) {
        List<NodeInfo> nodes = nodeService.getAllNodeIds();
        model.addAttribute("nodes", nodes);
        return "Home";
    }

    // RESTful API for React frontend
    @GetMapping("/api/nodes")
    @ResponseBody
    public List<NodeInfo> getNodes() {
        return nodeService.getAllNodeIds();
    }

    @GetMapping("/setPosition")
    @ResponseBody
    public String setPosition(@RequestParam double lat, @RequestParam double lng, @RequestParam double alt) {
        System.out.println("Received position: lat=" + lat + ", lng=" + lng + ", alt=" + alt);
        return "Position set: lat=" + lat + ", lng=" + lng + ", alt=" + alt;
    }
}
