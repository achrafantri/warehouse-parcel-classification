package com.example.achats.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * controlleur bech fel page d'acceuil yjina message.
 */
@RestController
public class HomeController {

    @GetMapping("/")
    public String home() {
        return "Bienvenue sur l'application Achats!";
    }
}
