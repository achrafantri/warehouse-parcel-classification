package com.example.achats.controller;

import com.example.achats.model.Fournisseur;
import com.example.achats.service.FournisseurService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/fournisseurs")
public class FournisseurController {
    @Autowired
    private FournisseurService service;

    @PostMapping
    public ResponseEntity<Fournisseur> createFournisseur(@RequestBody Fournisseur fournisseur) {
        service.ajouter(fournisseur);
        return ResponseEntity.ok(fournisseur);
    }

    @PutMapping("/{id}")
    public ResponseEntity<Void> updateFournisseur(@PathVariable Long id, @RequestBody Fournisseur fournisseur) {
        service.update(id, fournisseur);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/{id}")
    public Fournisseur rechercheParId(@PathVariable Long id) {
        return service.rechercheParId(id).orElse(null);
    }

    @GetMapping
    public List<Fournisseur> rechercheTout() {
        return service.rechercheTout();
    }

    @DeleteMapping("/{id}")
    public void supprimer(@PathVariable Long id) {
        service.supprimer(id);
    }
}
