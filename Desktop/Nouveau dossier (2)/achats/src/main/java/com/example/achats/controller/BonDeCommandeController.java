package com.example.achats.controller;

import com.example.achats.model.BonDeCommande;
import com.example.achats.service.BonDeCommandeService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/bons-de-commande")
public class BonDeCommandeController {

    @Autowired
    private BonDeCommandeService bonDeCommandeService;

    @GetMapping
    public List<BonDeCommande> getAllBonsDeCommande() {
        return bonDeCommandeService.findAll();
    }

    @PostMapping
public BonDeCommande createBonDeCommande(@RequestBody BonDeCommande bonDeCommande) {
        return bonDeCommandeService.save(bonDeCommande);
    }

    @GetMapping("/{id}")
    public ResponseEntity<BonDeCommande> getBonDeCommandeById(@PathVariable Long id) {
        BonDeCommande bonDeCommande = bonDeCommandeService.findById(id);
        return bonDeCommande != null ? ResponseEntity.ok(bonDeCommande) : ResponseEntity.notFound().build();
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteBonDeCommande(@PathVariable Long id) {
        bonDeCommandeService.deleteById(id);
        return ResponseEntity.noContent().build();
    }
}
