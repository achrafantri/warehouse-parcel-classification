package com.example.achats.controller;

import com.example.achats.model.Achat;
import com.example.achats.service.AchatService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;


import java.util.List;
import org.springframework.http.ResponseEntity;

@RestController
@RequestMapping("/api/achats")
public class AchatController {
    @Autowired
    private AchatService service;

    @PostMapping
    public ResponseEntity<Void> ajouter(@RequestBody Achat achat) {

        service.ajouter(achat);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/{id}")
    public Achat rechercheParId(@PathVariable Long id) {
        return service.rechercheParId(id).orElse(null);
    }

    @GetMapping
    public List<Achat> rechercheTout() {
        return service.rechercheTout();
    }

    @PutMapping("/{id}")
public ResponseEntity<Void> updateAchat(@PathVariable Long id, @RequestBody Achat achat) {
        service.update(id, achat);
        return ResponseEntity.ok().build();
    }

    @DeleteMapping("/{id}")
    public void supprimer(@PathVariable Long id) {
        service.supprimer(id);
    }
}
